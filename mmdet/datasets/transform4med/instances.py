"""
Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import numpy as np

from torch import Tensor
from typing import Dict, Union, Sequence, Tuple, Optional

from .base import AbstractTransform
from ..builder import PIPELINES
from .io4med import print_tensor, pdb, os
import random


@PIPELINES.register_module()
class FindInstances(AbstractTransform):
    """
    adopted from nnDetection
    finding the label value of instances in each sample (batch dimension)
    """

    def __init__(self, 
                instance_key: str = 'gt_instance_seg', 
                save_key: str = "present_instances", 
                meta_key = 'img_meta_dict', 
                map_key = 'inst2cls_map', 
                verbose = False, **kwargs):
        super().__init__(grad=False)
        self.instance_key = instance_key # target
        self.save_key = save_key
        self.verbose = verbose
        self.meta_key = meta_key
        self.map_key = map_key 

    def forward(self, data) -> dict:
        """ input data is a dict """
        present_instances = []
        # pdb.set_trace()
        # print(f'[meta_key] len' , len(data[self.meta_key]))
        # 1 means how many channels per chunk, split dim is defaulted to 0
        for bix, instance_element in enumerate(data[self.instance_key].split(1)):
            if self.verbose: print_tensor(f'\n[FindIns] {bix}', instance_element)
            tmp = instance_element.to(dtype=torch.int).unique(sorted=True)
            if tmp.max() > 64:
                print(f'[Instance] {tmp} exceed 64; case info is \n', data[self.meta_key][bix])
            tmp = tmp[tmp > 0]
            # inst2class may not contain instanceix found in the actual mask.
            if isinstance(self.map_key, str):
                inst2class_map = data[self.meta_key][bix][self.map_key]
                valid_instances = list(inst2class_map)
                valid_tmp_mask = [i for i, a in enumerate(tmp) if a.item() in valid_instances]
                if self.verbose: print(f'[FindInst] {bix} raw {tmp} validix {valid_tmp_mask}')
                tmp = tmp[valid_tmp_mask]
            present_instances.append(tmp)
        data[self.save_key] = present_instances
        return data

@PIPELINES.register_module()
class Instances2Boxes(AbstractTransform):
    def __init__(self, instance_key: str, map_key: str,
                 box_key: str, class_key: str, grad: bool = False,
                 present_instances: Optional[str] = None,
                 meta_key = 'img_meta_dict',  verbose = False,
                 move_jitter = 0, 
                 **kwargs):
        """
        Convert instance segmentation to bounding boxes

        Args
            instance_key: key where instance segmentation is located
            map_key: key where mapping from instances to classes is located
                (should be a dict which keys(instances) to items(classes))
            box_key: key where boxes should be saved
            class_key: key where classes of instances will be saved
            grad: enable gradient computation inside transformation
            present_instances: key where precomputed present instances are
                saved. If None it will compute the present instance new.
        """
        super().__init__(grad=grad, **kwargs)
        self.class_key = class_key
        self.box_key = box_key
        self.map_key = map_key
        self.instance_key = instance_key
        self.present_instances = present_instances
        self.meta_key = meta_key
        # self.just1cls = just1cls
        self.verbose = verbose
        self.move_jitter = move_jitter

    def forward(self, data) -> dict:
        """
        Extract boxes from instances

        Args:
            **data: batch dict

        Returns:
            dict: processed batch
        """
        data[self.box_key] = [] 
        data[self.class_key] = []
        for batch_idx, instance_element in enumerate(data[self.instance_key].split(1)):

            _present_instances = data[self.present_instances][batch_idx] if self.present_instances is not None else None
            _boxes_nx6, instance_idx = instances_to_boxes(instance_element, 
                                                      instance_element.ndim - 2, 
                                                      instances=_present_instances, 
                                                      jitter_base=self.move_jitter)
            inst2class_map = data[self.meta_key][batch_idx][self.map_key]
            if self.verbose: 
                print_tensor(f'\n[PatchBox] bix {batch_idx} instix {instance_idx} map {inst2class_map}', instance_element)
            # inst_in_map = [(str(int(a.detach().item())) not in inst2class_map) for a in instance_idx]
            # if any(inst_in_map): 
            #     print(data[self.meta_key][batch_idx]['filename_or_obj'])
            try: 
                _classes = get_instance_class_from_properties(instance_idx, inst2class_map) 
            except KeyError:
                _classes = torch.tensor([0 for on in instance_idx], device = instance_element.device)
                pid = data[self.meta_key][batch_idx]['filename_or_obj'].split(os.sep)[-1]
                print_tensor(f'[Inst2Box] key error pid {pid} sample tensor', instance_element )
            _classes = _classes.to(device=_boxes_nx6.device).long() # class start from 0
            # if self.just1cls is not None: _classes = self.just1cls
            if self.verbose: print_tensor(f'[Getbox] bx{batch_idx} ins2cls {inst2class_map} '
                                    + f'ins {instance_idx}  cls {_classes}, instmask', instance_element)
            data[self.box_key].append(_boxes_nx6)
            data[self.class_key].append(_classes)
        return data


def instances_to_boxes(seg: Tensor,
                       dim: int = None,
                       instances: Optional[Sequence[int]] = None,
                       jitter_base = 0, 
                       verbose = False,
                       ) -> Tuple[Tensor, Tensor]:
    """
    Convert instance segmentation to bounding boxes (not batched)

    Args
    seg: instance segmentation of individual classes [..., dims]
    dim: number of spatial dimensions to create bounding box for
        (always start from the last dimension). If None, all dimensions are
        used

    Returns
        Tensor: bounding boxes
            (x1, y1, x2, y2, (z1, z2)) List[Tensor[N, dim * 2]]
        Tensor: tuple with classes for bounding boxes
        #NOTE: could produce negative values in the gt bboxes 
    """
    if dim is None:
        dim = seg.ndim
    boxes = []
    _seg = seg.detach()

    if instances is None:
        instances = _seg.unique(sorted=True)
        instances = instances[instances > 0]

    for _idx in instances:
        instance_idx = (_seg == _idx).nonzero(as_tuple=False) # xyz

        _mins = instance_idx[:, -3:].min(dim=0)[0] # 
        _maxs = instance_idx[:, -3:].max(dim=0)[0]
        h1, w1, h2, w2 = _mins[-dim] - 1, _mins[(-dim) + 1] - 1, _maxs[-dim] + 1, _maxs[(-dim) + 1] + 1
        box = [h1, w1, h2, w2]
        if dim > 2:
            z1, z2 = [_mins[(-dim) + 2] - 1, _maxs[(-dim) + 2] + 1]
            box = [h1, w1, z1, h2, w2, z2]
        if verbose:
            print(f'[edge] min max bbox \n{_mins}\n{_maxs} \n{box}')
        box = add_jitter2bbox(box, jitter_base)
        boxes.append(torch.tensor(box))

    if boxes:
        boxes = torch.stack(boxes)
    else:
        boxes = torch.empty((0, 6))
    return boxes.to(dtype=torch.float, device=seg.device), instances


rand_in_range = lambda a, b : random.randrange(a, b)

def add_jitter2bbox(bbox, jitter_base = 3):
    """
    move bbox in each dimension with a small jitter
    not changing the size of the bbox
    """
    if jitter_base <= 0 : return bbox
    new_bbox = [0 for _ in bbox]
    for i in range(3):
        jitter = rand_in_range(-jitter_base, jitter_base)
        new_bbox[i], new_bbox[i + 3] = bbox[i] + jitter, bbox[i + 3] + jitter
    return new_bbox



def instances_to_boxes_np(
    seg: np.ndarray,
    dim: int = None,
    instances: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert instance segmentation to bounding boxes (not batched)

    Args
    seg: instance segmentation of individual classes [..., dims]
    dim: number of spatial dimensions to create bounding box for
        (always start from the last dimension). If None, all dimensions are
        used

    Returns
        np.ndarray: bounding boxes
            (x1, y1, x2, y2, (z1, z2)) List[Tensor[N, dim * 2]]
        np.ndarray: tuple with classes for bounding boxes
    """
    if dim is None:
        dim = seg.ndim
    boxes = []
    if instances is None:
        instances = np.unique(seg)
        instances = instances[instances > 0]

    for _idx in instances:
        instance_idx = np.stack(np.nonzero(seg == _idx), axis=1)
        _mins = np.min(instance_idx[:, -dim:], axis=0)
        _maxs = np.max(instance_idx[:, -dim:], axis=0)

        box = [_mins[-dim] - 1, _mins[(-dim) + 1] - 1, _maxs[-dim] + 1, _maxs[(-dim) + 1] + 1]
        if dim > 2:
            box = box + [_mins[(-dim) + 2] - 1, _maxs[(-dim) + 2] + 1]
        boxes.append(np.array(box))

    if boxes:
        boxes = np.stack(boxes)
    else:
        boxes = np.array([[]])
    return boxes, instances


def get_instance_class_from_properties(
        instance_idx: torch.Tensor, map_dict: Dict[str, Union[str, int]]) -> Tensor:
    """
    Extract instance classes form mapping dict

    Args:
        instance_idx: instance ids present in segmentaion
        map_dict: dict mapping instance ids (keys) to classes

    Returns:
        Tensor: extracted instance classes
    """
    instance_idx, _ = instance_idx.sort()
    classes = [int(map_dict[int(idx.detach().item())]) for idx in instance_idx]
    return torch.tensor(classes, device=instance_idx.device)


def get_instance_class_from_properties_seq(
        instance_idx: Sequence, map_dict: Dict[str, Union[str, int]]) -> Sequence:
    """
    Extract instance classes form mapping dict

    Args:
        instance_idx: instance ids present in segmentaion
        map_dict: dict mapping instance ids (keys) to classes

    Returns:
        Sequence[int]: extracted instance classes
    """
    instance_idx = sorted(instance_idx)
    classes = [int(map_dict[str(int(idx))]) for idx in instance_idx]
    return classes

@PIPELINES.register_module()
class Instances2SemanticSeg(AbstractTransform):
    def __init__(self, instance_key: str, map_key: str, seg_key: str = None,
                 add_background: bool = True, grad: bool = False,
                 present_instances: Optional[str] = None,
                 meta_key = 'img_meta_dict', verbose = False
                 ):
        """
        Convert instances to semantic segmentation

        Args:
            instance_key: key where instance segmentation is located
            map_key: key where mapping from instances to classes is located
            seg_key: key where segmentation should be saved; If None, the
                instance key will be overwritten
            add_background: adds +1 to classes from mapping for background
            grad: enable gradient propagation through transformation
            present_instances: key where precomputed present instances are
                saved. If None it will compute the present instance new.
        """
        super().__init__(grad=grad)
        self.add_background = add_background
        self.seg_key = seg_key if seg_key is not None else instance_key
        self.map_key = map_key
        self.instance_key = instance_key
        self.present_instances = present_instances
        self.meta_key = meta_key
        self.verbose = verbose

    def forward(self, data) -> dict:
        """
        Convert instance segmentation to semantic segmentation

        Args:
            **data: batch dict

        Returns:
            dict: processed batch
        """
        semantic = torch.zeros_like(data[self.instance_key])
        _present_instances = data[self.present_instances] if self.present_instances is not None else None
        for batch_idx in range(semantic.shape[0]):
            ins2cls = data[self.meta_key][batch_idx][self.map_key]
            ins_mask = data[self.instance_key][batch_idx]

            # TODO: check classes 
            instances_to_segmentation(ins_mask, ins2cls, 
                                      add_background=self.add_background,
                                      instance_idx=_present_instances[batch_idx],
                                      out=semantic[batch_idx])
            if self.verbose: 
                print_tensor(f'\n[Inst2Seg] bix {batch_idx} ins2cls {ins2cls} instance', ins_mask)
                print_tensor(f'[Inst2Seg] bix {batch_idx} ins2cls {ins2cls} semantic', semantic[batch_idx])
        data[self.seg_key] = semantic
        return data


def instances_to_segmentation(instances: Tensor,
                              mapping: Dict[str, Union[str, int]],
                              add_background: bool = True,
                              instance_idx: Optional[Sequence[int]] = None,
                              out: Tensor = None) -> Tensor:
    """
    Convert instances to semantic segmentation

    Args:
        instances: instance segmentation; foreground classes > 0; [dims]
        mapping: mapping from each instance to class
        add_background: adds +1 to classes from mapping for background
            Should be enabled if classes in mapping start from zero and
            diabled otherwise
        out: optional output tensor where results are saved
        instance_idx: precomputed instance ids present in sample. If None
            the instances ids will be computed

    Returns:
        Tensor: semantic segmentation
    """
    mapping = {int(key): int(item) for key, item in mapping.items()}
    if out is None:
        out = torch.zeros_like(instances)

    if instance_idx is None:
        instance_idx = instances.unique(sorted=True)
        instance_idx = instance_idx[instance_idx > 0]

    for instance_id in instance_idx:
        _cls = mapping[instance_id.item()]
        if add_background:
            _cls += 1
        out[instances == instance_id] = _cls
    return out


def instances_to_segmentation_np(instances: np.ndarray,
                                 mapping: Dict[Union[str, int], Union[str, int]],
                                 add_background: bool = True,
                                 out: np.ndarray = None) -> np.ndarray:
    """
    Convert instances to semantic segmentation

    Args:
        instances: instance segmentation; foreground classes > 0; [dims]
        mapping: mapping from each instance to class
        add_background: adds +1 to classes from mapping for background
            Should be enabled if classes in mapping start from zero and
            diabled otherwise
        out: optional output tensor where results are saved

    Returns:
        Tensor: semantic segmentation
    """
    mapping = {int(key): int(item) for key, item in mapping.items()}
    if out is None:
        out = np.zeros_like(instances)

    instance_idx = np.unique(instances)
    instance_idx = instance_idx[instance_idx > 0]
    for instance_id in instance_idx:
        _cls = mapping[instance_id]
        if add_background:
            _cls += 1
        out[instances == instance_id] = _cls
    return out


def get_bbox_np(seg: np.ndarray,
                map_dict: Optional[Dict[Union[str, int], Union[str, int]]] = None,
                **kwargs,
                ) -> dict:
    """
    Get bounding boxes and mapping from instances to classes
    
    Args:
        seg: instance segmentation [1, dims]
        mapping: define mapping from instance ids to classes
    
    Returns:
        dict: extracted boxes and classes
            `boxes` (np.ndarray): bounding boxes [N, dims * 2]
            `classes` (np.ndarray): classes (in same order as boxes) [N]
    """
    if map_dict is not None:
        map_dict = {str(key): str(item) for key, item in map_dict.items()}

    result = {}
    boxes, instance_idx = instances_to_boxes_np(seg[0], **kwargs)
    result["boxes"] = boxes

    if map_dict is not None:
        box_classes = get_instance_class_from_properties_seq(instance_idx, map_dict)
        result["classes"] = np.array(box_classes)
    return result
