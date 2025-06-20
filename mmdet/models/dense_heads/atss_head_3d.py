import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32, auto_fp16

from mmdet.core import (anchor_inside_flags_3d, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap, ATSSAssigner3D, HardNegPoolSampler)
from ..builder import HEADS, build_loss
from .anchor_head_3d import AnchorHead3D, chn2last_order, print_tensor
from mmdet.core.post_processing.bbox_nms import multiclass_nms_3d
import pdb


@HEADS.register_module()
class ATSSHead3D(AnchorHead3D): 
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv3d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='atss_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(ATSSHead3D, self).__init__(
            num_classes, in_channels, init_cfg=init_cfg, **kwargs)

        # self.sampling = True
        if self.train_cfg:
            self.assigner : ATSSAssigner3D = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler : HardNegPoolSampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        convnd = nn.Conv3d if self.spatial_dim == 3 else nn.Conv2d
        self.atss_cls = convnd(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = convnd(
            self.feat_channels, self.num_anchors * 6, 3, padding=1)
        self.atss_centerness = convnd(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

        print(f'[ATSSHead] spatial dim {self.spatial_dim} anchor/pixle {self.num_anchors}' )
        print(f'[ATSSHead] inchannels {self.in_channels} start level {self.start_level}')
        print(f'[ATSSHead] bbox scaler by level ', self.scales)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats[self.start_level:], self.scales)

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 6.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W, D).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        if self.verbose:
            print_tensor('\n[ATSS] forward input', x)
            print_tensor('[ATSS] forward class', cls_score)
            print_tensor('[ATSS] forward bbox', bbox_pred)
            print_tensor('[ATSS] forward centerness', centerness)
        return cls_score, bbox_pred, centerness

    def loss_single(self, anchors, cls_score, bbox_pred, centerness, labels,
                    label_weights, bbox_targets, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W, D).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 6, H, W, D).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 6).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 6.
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # pdb.set_trace()
        dim_reorder = chn2last_order(self.spatial_dim)
        # pdb.set_trace()
        # prediction 
        cls_score = cls_score.permute(*dim_reorder).reshape(-1, self.cls_out_channels).contiguous() ## BHWDx a2
        bbox_pred = bbox_pred.permute(*dim_reorder).reshape(-1, self.spatial_dim * 2) # BHWDx a6
        centerness = centerness.permute(*dim_reorder).reshape(-1) # prediction

        # priors and gt labels
        anchors = anchors.reshape(-1, self.spatial_dim * 2) # BA6; A = P*a
        bbox_targets = bbox_targets.reshape(-1, self.spatial_dim * 2)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        # compute_mask = label_weights > 0 #[compute_mask]
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # compute_labels= labels[compute_mask]
        # fg_gt_count = (compute_labels == 0).sum()
        # fg_weight_counts = compute_mask.sum()
        # print(f'\n[ATSSl] clsloss {loss_cls} compute_pixel:{fg_weight_counts}, '
        #         f'fg_pixel:{fg_gt_count} numsample{num_total_samples}')

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((labels >= 0)& (labels < bg_class_ind)).squeeze(1)
        
        # print_tensor(f'[Loss] fg count {pos_inds.sum()}', pos_inds)
        # pdb.set_trace()
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds] # nx6
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds] # nx6
            pos_centerness = centerness[pos_inds] 
            
            if self.verbose:
                print_tensor('[Centerness] pred ', pos_centerness)
                print_tensor('[Centerness] pos anchor', pos_anchors)
                print_tensor('[Centerness] pos bbox target', pos_bbox_targets)

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode( 
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)
        # pdb.set_trace()
        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W, D)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 6, H, W, D)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W, D)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 6) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-self.spatial_dim:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device) # outer list by image; inner list by level
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels, 
            cls_scores_list = cls_scores)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)
        # pdb.set_trace()
        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

    def centerness_target(self, anchors, bbox_targets, verbose = False):
        # only calculate pos centerness targets, otherwise there may be nan
        # range from 0~1, 1 means the predicted bbox center is right on the center of gt
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        bbox2point = lambda nx: nx.view(nx.shape[0], 2, -1).float().mean(dim = 1) 
        anchor_points_nx3 = bbox2point(anchors)
        center2small_bound = anchor_points_nx3 - gts[:, :3]
        center2large_bound = gts[:, 3:] - anchor_points_nx3
        dist_bound_nx3x2 = torch.stack([center2small_bound, center2large_bound], dim = -1).clamp(min = 1e-4)
        centerness = torch.sqrt(
            (dist_bound_nx3x2[:, 0].min(dim=-1)[0] / dist_bound_nx3x2[:, 0].max(dim=-1)[0]) *
            (dist_bound_nx3x2[:, 1].min(dim=-1)[0] / dist_bound_nx3x2[:, 1].max(dim=-1)[0]) *
            (dist_bound_nx3x2[:, 2].min(dim=-1)[0] / dist_bound_nx3x2[:, 2].max(dim=-1)[0])
             )
        if verbose:
            print_tensor('[CerterNess] anchors', anchors)
            print_tensor('[CerterNess] bbox targets', bbox_targets)
            print_tensor('[CerterNess] bbox gt', gts)
            print_tensor('[CenterNess] map', centerness)
            # pdb.set_trace()
        
        assert not torch.isnan(centerness).any()
        return centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-self.spatial_dim:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device) 

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        img_shapes = [
            img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        # pdb.set_trace()
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_anchors,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 6).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            
            if self.verbose:
                print_tensor(f'\n[Pred2Bbox] anchors raw', anchors)
                print_tensor(f'[Pred2Bbox] pred score raw', cls_score)
                print_tensor(f'[Pred2Bbox] pred bbox raw', bbox_pred)
                print_tensor(f'[Pred2Bbox] pred center raw', centerness)

            assert cls_score.size()[-self.spatial_dim:] == bbox_pred.size()[-self.spatial_dim:]
            dim_reorder = chn2last_order(self.spatial_dim)
            scores = cls_score.permute(*dim_reorder).reshape(
                                batch_size, -1, self.cls_out_channels)
            
            scores = scores.sigmoid() if self.use_sigmoid_cls else scores.softmax(-1)

            centerness = centerness.permute(*dim_reorder).reshape(
                                                batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(*dim_reorder).reshape(
                                                batch_size, -1, self.spatial_dim * 2)

            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor, nms_pre_tensor, num_anchor)
                max_scores, max_ix = (centerness[..., None] *  (scores if self.use_sigmoid_cls else scores[..., :-1])).max(-1) #
                # max_scores_full, max_ix_full = (centerness[..., None] * scores).max(-1) #
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]
            else:
                anchors = anchors.expand_as(bbox_pred)

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            if self.verbose:
                print_tensor(f'[Pred2Bbox] top {nms_pre} bboxes ', bboxes)
                print_tensor(f'[Pred2Bbox] top {nms_pre} scores', scores)
                print_tensor(f'[Pred2Bbox] top {nms_pre} centerness', centerness)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                batch_mlvl_scores *
                batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds, topk_inds]
        if self.use_sigmoid_cls: 
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                batch_mlvl_scores.shape[1], 1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms_3d(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness)

                if self.verbose and det_bbox.shape[0]> 0:
                    print_tensor(f'\t[Pred2Bbox] NMS bbox ', det_bbox[:, :6])
                    print_tensor(f'\t[Pred2Bbox] NMS score', det_bbox[:, 6])
                    print_tensor(f'\t[Pred2Bbox] NMS label', det_label)
                    # print_tensor(f'\t[Pred2Bbox] NMS centerness', det_label)

                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    cls_scores_list = None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        grid2nx2 = lambda x, cls: x.permute(1, 2, 3, 0).reshape(-1, cls)
        concat_cls_score_list = [] 
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
            cls_score_img = torch.cat([grid2nx2(cls_scores_list[l][i], self.cls_out_channels) 
                                        for l in range(len(num_level_anchors))], dim = 0)
            concat_cls_score_list.append(cls_score_img)
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             concat_cls_score_list, 
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           cls_scores, 
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags_3d(flat_anchors, valid_flags,
                                           img_meta['img_meta_dict']['patch_shape'][:self.spatial_dim], 
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        cls_scores = cls_scores[inside_flags, : ]

        if self.verbose and gt_bboxes.shape[0] > 0: 
            print_tensor('\n[GetTarget] anchors', anchors)
            print_tensor('[GetTarget] cls scores', cls_scores)
            print_tensor('[GetTarget] gt bbox', gt_bboxes)
            print_tensor('[GetTarget] gt labels', gt_labels)

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes, cls_scores = cls_scores) 
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ), self.num_classes,  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # pdb.set_trace()
        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
