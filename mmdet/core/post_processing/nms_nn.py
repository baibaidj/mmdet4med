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
from torch import Tensor
from torch.cuda.amp import autocast
from torchvision.ops.boxes import nms as nms_2d

from ..utils._C import nms as nms_3d
# from nndet.core.boxes.ops import box_iou
from ..bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d as box_iou
import pdb

def nms_cpu(boxes, scores, thresh):
    """
    Performs non-maximum suppression for 3d boxes on cpu
    
    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded
    
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    ious = box_iou(boxes, boxes)
    _, _idx = torch.sort(scores, descending=True)
    
    keep = []
    while _idx.nelement() > 0:
        keep.append(_idx[0])
        # get all elements that were not matched and discard all others.
        non_matches = torch.where((ious[_idx[0]][_idx] <= thresh))[0]
        _idx = _idx[non_matches]
    return torch.tensor(keep).to(boxes).long()


@autocast(enabled=False)
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    Performs non-maximum suppression
    
    Args:
        boxes (Tensor): tensor with boxes (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): score for each box [N]
        iou_threshold (float): threshould when boxes are discarded
    
    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    if boxes.shape[1] == 4:
        # prefer torchvision in 2d because they have c++ cpu version
        nms_fn = nms_2d
    else:
        if boxes.is_cuda:
            nms_fn = nms_3d
        else:
            nms_fn = nms_cpu
    return nms_fn(boxes.float(), scores.float(), iou_threshold)


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float = 0.6):
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    
    Args:
        boxes (Tensor): boxes where NMS will be performed. (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
        scores (Tensor): scores for each one of the boxes [N]
        idxs (Tensor): indices of the categories for each one of the boxes. [N]
        iou_threshold (float):  discards all overlapping boxes with IoU > iou_threshold
    
    Returns
        keep (Tensor): int64 tensor with the indices of the elements that have been kept by NMS, 
            sorted in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)

def xyz2xy1z(coords_nx6):
    # from x1y1z1x2y2z2 to x1x2y1y2z1z2
    assert len(coords_nx6.shape) == 2 and coords_nx6.shape[-1] == 6
    return coords_nx6[..., [0, 1, 3, 4, 2, 5]]

def xy1z2xyz(coords_nx6):
    # from x1x2y1y2z1z2 to x1y1z1x2y2z2
    assert len(coords_nx6.shape) == 2 and coords_nx6.shape[-1] == 6
    return coords_nx6[..., [0, 1, 4, 2, 3, 5]]

def batched_nms_3d(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 6).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
    # compatible to the NMS operation developed by nndet
    boxes_for_nms = xyz2xy1z(boxes_for_nms)
    # nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = eval(nms_type)
    # split_thr = nms_cfg_.pop('split_thr', 10000)

    max_num = nms_cfg_.pop('max_num', -1)
    total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    # Some type of nms would reweight the score, such as SoftNMS
    scores_after_nms = scores.new_zeros(scores.size())
    for id in torch.unique(idxs):
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        # NOTE: nms 3d applied here
        # pdb.set_trace()
        keep = nms_3d(boxes_for_nms[mask], scores[mask], nms_cfg_['iou_threshold'])
        total_mask[mask[keep]] = True
        scores_after_nms[mask[keep]] = scores[mask[keep]]
    keep = total_mask.nonzero(as_tuple=False).view(-1)

    scores, inds = scores_after_nms[keep].sort(descending=True)
    keep = keep[inds]
    boxes = boxes[keep]

    if max_num > 0:
        keep = keep[:max_num]
        boxes = boxes[:max_num]
        scores = scores[:max_num]

    return torch.cat([boxes, scores[:, None]], -1), keep