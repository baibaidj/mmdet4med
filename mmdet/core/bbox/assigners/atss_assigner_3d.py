import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.utils import print_tensor 
import pdb


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner3D(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
        center_within (boolean) : nndetection suggest not to use this requirement
    """

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps3D'),
                 ignore_iof_thr=-1, 
                 center_within = True,
                 verbose = False):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr
        self.center_within = center_within
        self.verbose = verbose

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 6).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 6).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        overlaps_nxg = self.iou_calculator(bboxes, gt_bboxes) # nxg

        # assign 0 by default
        assigned_gt_inds = overlaps_nxg.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps_nx1 = overlaps_nxg.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps_nxg.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult( 
                num_gt, assigned_gt_inds, max_overlaps_nx1, labels=assigned_labels)

        # compute center distance between all bbox and gt
        bbox2point = lambda nx: nx.view(nx.shape[0], 2, -1).float().mean(dim = 1) 
        gt_points = bbox2point(gt_bboxes) # gx3
        bboxes_points = bbox2point(bboxes) # nx3

        distances_nxg = (bboxes_points[:, None, :] - # nx1x3, 1xgx3 = nxgx3
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt() # nxg

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances_nxg[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances_nxg[start_idx:end_idx, :] # mxg
            selectable_k = min(self.topk, bboxes_per_level)
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx) # kxg
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0) # klxg

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps_nxg[candidate_idxs, torch.arange(num_gt)] # klxg
        overlaps_mean_per_gt = candidate_overlaps.mean(0) # 1xg
        overlaps_std_per_gt = candidate_overlaps.std(0) # 1xg
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt  # 1xg

        is_pos_klxg = candidate_overlaps >= overlaps_thr_per_gt[None, :] # klxg, binary
        # pdb.set_trace()
        # To accommodate the flattening in 
        for gt_idx in range(num_gt): candidate_idxs[:, gt_idx] += gt_idx * num_bboxes 
        # limit the positive sample's center in gt
        if self.center_within:
            candidate_idxs = candidate_idxs.view(-1)
            trans_center_dim = lambda p, dim: p[:, dim].view(1, -1).expand( # 1xn > kxn > kn
                                        num_gt, num_bboxes).contiguous().view(-1)
            ep_bboxes_cx = trans_center_dim(bboxes_points, 0)
            ep_bboxes_cy = trans_center_dim(bboxes_points, 1)
            ep_bboxes_cz = trans_center_dim(bboxes_points, 2)

            # calculate the left, top, anterior, right, bottom, posterior distance between positive
            # bbox center and gt side
            l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
            t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
            a_ = ep_bboxes_cz[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 2]
            r_ = gt_bboxes[:, 3] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
            b_ = gt_bboxes[:, 4] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
            p_ = gt_bboxes[:, 5] - ep_bboxes_cz[candidate_idxs].view(-1, num_gt)
            # pdb.set_trace()
            is_in_gts = torch.stack([l_, t_, a_, r_, b_, p_], dim=1).min(dim=1)[0] > 0.01
            is_pos_klxg = is_pos_klxg & is_in_gts
        
        if self.verbose:
            print('[ATSSAssigner] positive candidate in GT matrix \n ', is_pos_klxg)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf_gxn = torch.full_like(overlaps_nxg, # nxg > gxn
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos_klxg.view(-1)] # 
        overlaps_inf_gxn[index] = overlaps_nxg.t().contiguous().view(-1)[index] # gxn
        if self.verbose:
            print_tensor('[Assigner] positive bbox overlap', overlaps_inf_gxn[index])
        overlaps_inf_nxg = overlaps_inf_gxn.view(num_gt, -1).t() # gn > gxn> nxg

        max_overlaps_nx1, argmax_overlaps_nx1 = overlaps_inf_nxg.max(dim=1)
        assigned_gt_inds[
            max_overlaps_nx1 != -INF] = argmax_overlaps_nx1[max_overlaps_nx1 != -INF] + 1
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
        
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps_nx1, labels=assigned_labels)
