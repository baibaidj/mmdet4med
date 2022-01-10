# Copyright (c) OpenMMLab. All rights reserved.
import sys
from inspect import signature

import torch, pdb
from mmcv.ops import batched_nms
from mmdet.core import (bbox, bbox_mapping_back, merge_aug_proposals, multiclass_nms, 
                        bbox_mapping_back_3d, multiclass_nms_3d)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin(object):
    """Mixin class for testing det bboxes via DenseHead."""

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=False)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            if len(bbox_outputs) >= 3:
                aug_labels.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.test_cfg.nms)
        det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.test_cfg.max_per_img]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        """Test with augmentation for only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals

    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self(x)

            proposal_list = self.get_bboxes(*rpn_outs, img_metas=img_metas)
            return proposal_list

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores



class BBoxTestMixin3D(object):
    """Mixin class for testing det bboxes via DenseHead."""

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 5D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 7),
                where 57represent (tl_x, tl_y, tl_z, br_x, br_y, br_z, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        # TODO: put tile_origin into the img_metas for bbox offset
        results_list = reset_offset_bbox_batch(results_list, img_metas)
        return results_list

    # def aug_test_bboxes(self, feats, img_metas, rescale=False):
    #     """Test det bboxes with test time augmentation, can be applied in
    #     DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
    #     etc.

    #     Args:
    #         feats (list[Tensor]): the outer list indicates test-time
    #             augmentations and inner Tensor should have a shape NxCxHxWxD,
    #             which contains features for all images in the batch.
    #         img_metas (list[list[dict]]): the outer list indicates test-time
    #             augs (multiscale, flip, etc.) and the inner list indicates
    #             images in a batch. each dict has image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is ``bboxes`` with shape (n, 7),
    #             where 7 represent (tl_x, tl_y, tl_z, br_x, br_y, br_z, score).
    #             The shape of the second tensor in the tuple is ``labels``
    #             with shape (n,). The length of list should always be 1.
    #     """
    #     # check with_nms argument
    #     gb_sig = signature(self.get_bboxes)
    #     gb_args = [p.name for p in gb_sig.parameters.values()]
    #     if hasattr(self, '_get_bboxes'):
    #         gbs_sig = signature(self._get_bboxes)
    #     else:
    #         gbs_sig = signature(self._get_bboxes_single)
    #     gbs_args = [p.name for p in gbs_sig.parameters.values()]
    #     assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
    #         f'{self.__class__.__name__}' \
    #         ' does not support test-time augmentation'

    #     aug_bboxes = []
    #     aug_scores = []
    #     aug_factors = []  # score_factors for NMS
    #     for x, img_meta in zip(feats, img_metas):
    #         # only one image in the batch
    #         outs = self.forward(x) # no NMS
    #         bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
    #         bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            
    #         aug_bboxes.append(bbox_outputs[0])
    #         aug_scores.append(bbox_outputs[1])
    #         # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
    #         # contains additional element to adjust scores before NMS
    #         if len(bbox_outputs) >= 3:
    #             aug_factors.append(bbox_outputs[2])

    #     # after merging, bboxes will be rescaled to the original image size
    #     # recover flip and resize
    #     merged_bboxes, merged_scores = self.merge_aug_bboxes(
    #         aug_bboxes, aug_scores, img_metas)
    #     merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
    #     det_bboxes, det_labels = multiclass_nms_3d(
    #         merged_bboxes,
    #         merged_scores,
    #         self.test_cfg.score_thr,
    #         self.test_cfg.nms,
    #         self.test_cfg.max_per_img,
    #         score_factors=merged_factors)

    #     if rescale:
    #         _det_bboxes = det_bboxes
    #     else:
    #         _det_bboxes = det_bboxes.clone()
    #         _det_bboxes[:, :6] *= det_bboxes.new_tensor(
    #             img_metas[0][0]['scale_factor'])

    #     return [
    #         (_det_bboxes, det_labels),
    #     ]

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        """Test with augmentation for only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals

    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self(x)

            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
            return proposal_list

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 6*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,6), where
            6 represent (tl_x, tl_y, tl_z, br_x, br_y, br_z)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back_3d(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

from torch import Tensor
from typing import Dict, Sequence
# from mmdet.core.bbox.ops import box_center

@torch.no_grad()
def reset_offset_bbox_batch(results, img_metas):
    """
    Process a single batch of bounding box predictions
    (the boxes are clipped to the case size in the ensembling step)

    Args:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                    The first item is ``bboxes`` with shape (n, 7),
                    where 57represent (tl_x, tl_y, tl_z, br_x, br_y, br_z, score).
                    The shape of the second tensor in the tuple is ``labels``
                    with shape (n,)

        result: prediction from detector. Need to provide boxes, scores
            and class labels
                `self.box_key`: List[Tensor]: predicted boxes (relative
                    to patch coordinates)
                `self.score_key` List[Tensor]: score for each tensor
                `self.label_key`: List[Tensor] label prediction for each box
        batch: input batch for detector
            `tile_origin: origin of crop with respect to actual data (
                in case of padding)
            `crop`: Sequence[slice] original crop from data
    """
    assert len(results) == len(img_metas)

    for i, (img_result, img_meta) in enumerate(zip(results, img_metas)):
        det_box_nx7, det_label = img_result
        tile_origin  = img_meta.get('tile_origin', None)
        if tile_origin is None: continue
        det_box_nx7_new = _apply_offsets_to_boxes(det_box_nx7, tile_origin)
        results[i] = tuple([det_box_nx7_new, det_label])

    return results

def _apply_offsets_to_boxes(img_boxes: Tensor, offset: Sequence[int],
                                ) -> Tensor:
    """
    Apply offset to bounding boxes to position them correctly inside
    the whole case

    Args:
        boxes: Tensor predicted boxes [N, dims * 2 + 1]
            [x1, y1, x2, y2, (z1, z2))
        tile_offset: defines offset for each tile

    Returns:
        Tensor: bounding boxes with respect to origin of whole case
    """
    if img_boxes.nelement() == 0:
        return img_boxes
    offset = img_boxes.new_tensor(offset)
    img_boxes[:, [0, 3]] += offset[0]
    img_boxes[:, [1, 4]] += offset[1]
    img_boxes[:, [2, 5]] += offset[2]
    return img_boxes



def _get_box_in_tile_weight(box_centers: Tensor,
                            tile_size: Sequence[int],
                            ) -> Tensor:
    """
    Assign boxes near the corner a lower weight.
    The midle has a plateau with weight one, starting from patchsize / 2
    the weights decreases linearly until 0.5 is reached. 

    Args:
        box_centers: center predicted box [N, dims]
        tile_size: size the of patch/tile

    Returns:
        Tensor: weight for each bounding box [N]
    """
    plateau_length = 0.5  # adjust width of plateau and min weight
    if box_centers.numel() > 0:
        tile_center = torch.tensor(tile_size).to(box_centers) / 2.  # [dims]

        max_dist = tile_center.norm(p=2)  # [1]
        boxes_dist = (box_centers - tile_center[None]).norm(p=2, dim=1) # [N]
        weight = -(boxes_dist / max_dist - plateau_length).clamp_(min=0) + 1
        return weight
    else:
        return Tensor([]).to(box_centers)