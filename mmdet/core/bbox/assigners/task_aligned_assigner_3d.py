import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import TaskAlignedAssignResult
from .base_assigner import BaseAssigner
from mmdet.utils import print_tensor
import ipdb


@BBOX_ASSIGNERS.register_module()
class TaskAlignedAssigner3D(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
        alpha (float):  weight for classification scores, default to 1
        beta (float): weight for IoU values, default to 6
    """

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps3D'),
                 ignore_iof_thr=-1, 
                 alpha=1,
                 beta=6, 
                 **kwargs):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.alpha = alpha
        self.beta = beta

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 6), "xyzxyz" format.

        Returns:
            Tensor: Anchor centers with shape (N, 3), "xyz" format.
        """
        anchors_cx = (anchors[:, 3] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 4] + anchors[:, 1]) / 2
        anchors_cz = (anchors[:, 5] + anchors[:, 2]) / 2
        return torch.stack([anchors_cx, anchors_cy, anchors_cz], dim=-1)

    def assign(self,
               scores,
               decode_bboxes,
               anchors,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector only can predict positive distance)


        Args:
            scores (Tensor): predicted class probability, shape(n, 80)
            decode_bboxes (Tensor): predicted bounding boxes, shape(n, 6)
            anchors (Tensor): pre-defined anchors, shape(n, 6).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 6).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`TaskAlignedAssignResult`: The assign result.
        """
        INF = 100000000
        anchors = anchors[:, :6]
        num_gt, num_bboxes = gt_bboxes.size(0), anchors.size(0)

        # compute alignment metric between all bbox and gt
        overlaps_nxg = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = scores[:, gt_labels].detach()
        alignment_metrics_nxg = bbox_scores ** self.alpha * overlaps_nxg ** self.beta
        # print_tensor('overlap nxg', overlaps_nxg)
        # print_tensor('bbox score ', bbox_scores)
        # print_tensor('align metric', alignment_metrics_nxg)
        # assign 0 by default
        assigned_gt_inds = alignment_metrics_nxg.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assign_metrics = alignment_metrics_nxg.new_zeros((num_bboxes, ))

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps_nx1 = overlaps_nxg.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = alignment_metrics_nxg.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return TaskAlignedAssignResult(
                num_gt, assigned_gt_inds, max_overlaps_nx1, assign_metrics, labels=assigned_labels)

        # select top-k bbox as candidates for each gt
        _, candidate_idxs = alignment_metrics_nxg.topk(
            self.topk, dim=0, largest=True)
        candidate_metrics = alignment_metrics_nxg[candidate_idxs, torch.arange(num_gt)]
        is_pos_klxg = candidate_metrics > 0

        # if require positive sample located within center
        bbox2point = lambda nx: nx.view(nx.shape[0], 2, -1).float().mean(dim = 1) 
        anchors_points = bbox2point(anchors) # nx3
        candidate_bboxes = anchors_points[candidate_idxs, :] # klxgx3
        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = candidate_bboxes[:, :, 0] - gt_bboxes[:, 0][None, :]  # klxg - 1xg
        t_ = candidate_bboxes[:, :, 1] - gt_bboxes[:, 1][None, :]  # klxg - 1xg
        a_ = candidate_bboxes[:, :, 2] - gt_bboxes[:, 2][None, :]  # klxg - 1xg
        r_ = gt_bboxes[:, 3][None, :] - candidate_bboxes[:, :, 0]  # 1xg - klxg
        b_ = gt_bboxes[:, 4][None, :] - candidate_bboxes[:, :, 1]  # 1xg - klxg
        p_ = gt_bboxes[:, 5][None, :] - candidate_bboxes[:, :, 2]  # 1xg - klxg
        is_in_gts = torch.stack([l_, t_, a_, r_, b_, p_], dim=1).min(dim=1)[0] >= 0 
        is_pos_klxg = is_pos_klxg & is_in_gts

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt): candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        overlaps_inf_gxn = torch.full_like(overlaps_nxg,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs[is_pos_klxg.view(-1)]
        overlaps_inf_gxn[index] = overlaps_nxg.t().contiguous().view(-1)[index]
        overlaps_inf_nxg = overlaps_inf_gxn.view(num_gt, -1).t() # gn > gxn> nxg

        max_overlaps_nx1, argmax_overlaps_nx1 = overlaps_inf_nxg.max(dim=1)
        fg_mask = max_overlaps_nx1 != -INF
        assigned_gt_inds[fg_mask] = argmax_overlaps_nx1[fg_mask] + 1
        assign_metrics[fg_mask] = alignment_metrics_nxg[fg_mask, argmax_overlaps_nx1[fg_mask]]

        # In assigned_gt_inds, 0 stands for background, so the gt index has to start from 1. 
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return TaskAlignedAssignResult(
            num_gt, assigned_gt_inds, max_overlaps_nx1, assign_metrics, labels=assigned_labels)
