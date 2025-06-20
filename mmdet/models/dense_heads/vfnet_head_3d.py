# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from numpy.lib import stride_tricks
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
# from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32

from mmdet.core import (bbox2distance3d, bbox_overlaps_3d, build_prior_generator,
                        build_assigner, build_sampler, distance2bbox3d,
                        multi_apply, multiclass_nms_3d, reduce_mean, 
                        HardNegPoolSampler, ATSSAssigner3D)
from ..builder import HEADS, build_loss
from .atss_head_3d_noc import ATSSHead3DNOC, print_tensor
from .fcos_head_3d import FCOSHead3D
from dcn import DeformConv as DeformConv3D
import pdb

INF = 1e8

@HEADS.register_module()
class VFNetHead3D(ATSSHead3DNOC, FCOSHead3D):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 16), (16, 32), (32, 64), (64, 128),
                                 (128, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 use_atss=True,
                 anchor_generator=dict(
                     type='AnchorGenerator3D',
                     ratios=[1.0],
                     octave_base_scale=4,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[4, 8, 16, 32]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv3d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 27
        self.dcn_kernel = int(np.power(self.num_dconv_points, 1/3)) # 3 
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = torch.arange(-self.dcn_pad,self.dcn_pad + 1, dtype=torch.long)
        self.dcn_offset_nx3 =  torch.stack(torch.meshgrid([dcn_base, dcn_base, dcn_base]), 
                                            axis = -1).reshape(-1, 3)
        # print_tensor('DCN offset nx3', self.dcn_offset_nx3)
        self.dcn_base_offset = self.dcn_offset_nx3.view(1, -1, 1, 1, 1)

        super(FCOSHead3D, self).__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        regress_ranges = [regress_ranges[i] for i, s in enumerate(self.strides)]
        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

        # for getting ATSS targets
        self.use_atss = use_atss
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_generator = build_prior_generator(anchor_generator)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.sampling = False
        if self.train_cfg:
            self.assigner : ATSSAssigner3D = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler : HardNegPoolSampler = build_sampler(sampler_cfg, context=self)

    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead3D, self)._init_cls_convs()
        super(FCOSHead3D, self)._init_reg_convs()
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv3d(self.feat_channels, 6, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv3D( # NOTE: 
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv3d(self.feat_channels, 6, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv3D(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv3d(self.feat_channels, self.cls_out_channels, 3, padding=1)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        return multi_apply(self.forward_single, feats[self.start_level:], self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x, scale, scale_refine, stride, reg_denom):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        bbox_pred_raw = scale(self.vfnet_reg(reg_feat_init))
        
        # print_tensor('\nclsfeat', cls_feat)
        # print_tensor('regfeat', reg_feat)
        # print_tensor('regfeat init', reg_feat_init)
        # print_tensor('bbox pred raw', bbox_pred_raw)
        # pdb.set_trace()

        with torch.cuda.amp.autocast(enabled = False):
            if self.bbox_norm_type == 'reg_denom':
                bbox_pred = bbox_pred_raw.float().exp() * reg_denom
            elif self.bbox_norm_type == 'stride':
                bbox_pred = bbox_pred_raw.float().exp() * stride
            else:
                raise NotImplementedError

            # print_tensor('bbox_pred', bbox_pred)
            # compute star deformable convolution offsets
            # converting dcn_offset to reg_feat.dtype thus VFNet can be
            # trained with FP16
            dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul,
                                            stride).to(bbox_pred.dtype)
            # print_tensor('dcn offset', dcn_offset)
            # refine the bbox_pred
            reg_feat_fp32 = reg_feat.float()
            reg_feat_fp32 = self.relu(self.vfnet_reg_refine_dconv(reg_feat_fp32, dcn_offset))
            bbox_pred_refine = scale_refine(self.vfnet_reg_refine(reg_feat_fp32)).float().exp()
            # predict the iou-aware cls score
            cls_feat = self.vfnet_cls_dconv(cls_feat.float(), dcn_offset)
        
        # print_tensor('bbox pred refine', bbox_pred_refine)
        # print_tensor('bbox_pred raw', bbox_pred_raw)

        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()
        cls_score = self.vfnet_cls(self.relu(cls_feat))

        # print_tensor('cls feat refine', cls_feat)
        # print_tensor('cls score', cls_score)
        return cls_score, bbox_pred, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.
        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (t, l, a, b, r, p).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred) # (1, 81, 1, 1, 1), 27 points
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W, D = bbox_pred.size()

        t_h1 = bbox_pred_grad_mul[:, 0, :, :] # top
        l_w1 = bbox_pred_grad_mul[:, 1, :, :] # left
        a_d1 = bbox_pred_grad_mul[:, 2, :, :] # anterior
        b_h2 = bbox_pred_grad_mul[:, 3, :, :] # bottom
        r_w2 = bbox_pred_grad_mul[:, 4, :, :] # right
        p_d2 = bbox_pred_grad_mul[:, 5, :, :] # posterior
 
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros( N, 3 * self.num_dconv_points, H, W, D) # 3*9 

        for i, point3 in enumerate(self.dcn_offset_nx3):
            h_sign, w_sign, d_sign = point3 # (-1, -1, -1)
            h_ix, w_ix, d_ix= i * 3, (i * 3 + 1), (i*3 + 2)
            if h_sign < 0:  h_offset = h_sign * t_h1 
            elif h_sign > 0: h_offset = h_sign * b_h2
            else: h_offset = None

            if w_sign < 0: w_offset= w_sign * l_w1
            elif w_sign > 0: w_offset = w_sign * r_w2
            else: w_offset = None

            if d_sign < 0: d_offset = d_sign * a_d1
            elif d_sign > 0: d_offset = d_sign * p_d2
            else: d_offset = None
            
            if h_offset is not None:
                bbox_pred_grad_mul_offset[:, h_ix, ...] = h_offset
            if w_offset is not None:
                bbox_pred_grad_mul_offset[:, w_ix, ...] = w_offset
            if d_offset is not None:
                bbox_pred_grad_mul_offset[:, d_ix, ...] = d_offset
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset
        return dcn_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 5D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 5D-tensor, the channel number is
                num_points * 6.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 6.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 6) in [tl_x, tl_y, tl_z, br_x, br_y, br_z] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        # if self.verbose: 
        #     print_tensor(f'\n[AnchorPoint] dim0 coord range', all_level_points[0][:, 0])
        #     print_tensor(f'[AnchorPoint] dim1 coord range', all_level_points[0][:, 1])
        #     print_tensor(f'[AnchorPoint] dim2 coord range', all_level_points[0][:, 2])

        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 4, 1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, 6).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 4, 1).reshape(-1, 6).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = distance2bbox3d(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = distance2bbox3d(pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps_3d(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = \
            distance2bbox3d(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps_3d(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        if self.verbose: 
            print_tensor('[VFHead] pos target distance', pos_bbox_targets)
            print_tensor('[VFHead] pos preds distance', pos_bbox_preds)
            print_tensor('[VFHead] pos target bbox', pos_decoded_target_preds)
            print_tensor('[VFHead] pos preds bbox', pos_decoded_bbox_preds)
            print_tensor('[VFHead] IOU initial', iou_targets_ini)
            print_tensor('[VFHead] IOU refine', iou_targets_rf)

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)

        if self.verbose: 
            print_tensor(f'[VFLoss] USEVF-{self.use_vfl} cls score', flatten_cls_scores)
            print_tensor('[VFLoss] cls label target', flatten_labels)
            print_tensor(f'[VFLoss] cls iou target', cls_iou_targets)
            print(f'[VFLoss] cls loss {loss_cls} bbox loss {loss_bbox}')
            # print_tensor('[VFloss] label weight', label_weights)
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   with_nms=True):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W, D).
            bbox_preds (list[Tensor]): Box offsets for each scale
                level with shape (N, num_points * 6, H, W, D).
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level with shape (N, num_points * 6, H, W, D).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list, mlvl_points,
                                                 img_shape, scale_factor, cfg,
                                                 rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for a single scale
                level with shape (num_points * num_classes, H, W, D).
            bbox_preds (list[Tensor]): Box offsets for a single scale
                level with shape (num_points * 6, H, W, D).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 6).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 7), where
                    the first 6 columns are bounding box positions
                    (tl_x, tl_y, tl_z, br_x, br_y, br_z) and the 7-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds,
                                                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 3, 0).reshape(
                -1, self.cls_out_channels).contiguous().sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(-1, 6).contiguous()

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox3d(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            det_bboxes, det_labels = multiclass_nms_3d(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w, d = featmap_size
        w_range = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        h_range = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        d_range = torch.arange(0, d * stride, stride, dtype=dtype, device=device)
        hh, ww, dd = torch.meshgrid([h_range, w_range, d_range])
        # to be compatible with anchor points in ATSS
        if self.use_atss:
            points = torch.stack( # TODO: here the order of ww and hh is incompatible with that of hw
                (hh.reshape(-1), ww.reshape(-1), dd.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset
        else:
            points = torch.stack(
                (hh.reshape(-1), ww.reshape(-1), dd.reshape(-1)), dim=-1) + \
                    stride // 2
        return points

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels,
                    img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W, D).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 3).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 6).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 6).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead3DNOC._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead3D._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead3D.get_targets(self, points,
                                                    gt_bboxes_list,
                                                    gt_labels_list)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_atss_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W, D).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 3).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 6).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 6). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (top, left, anterior, bottom, right, poterior).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = ATSSHead3DNOC.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            cls_scores_list=cls_scores, 
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        # Check if anchor position is equivalent to point position
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 6) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)

        # transform bbox_targets (h1, w1, d1, h2, w2, d2) into (t, l, a, b, r, p) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, z1, x2, y2, z2) into (l, t, anteiror, r, b, posterior) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, z1, x2, y2, z2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 3).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, anteiror, r, b, posterior).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = bbox2distance3d(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass
