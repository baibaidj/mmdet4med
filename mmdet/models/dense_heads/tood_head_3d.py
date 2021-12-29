import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
import torch.nn.functional as F

from mmdet.core import (anchor_inside_flags_3d, build_assigner, build_sampler,
                        images_to_levels, multi_apply,
                        reduce_mean, unmap,
                        distance2bbox3d, multiclass_nms_3d, 
                        ATSSAssigner3D, TaskAlignedAssigner3D)
from ..builder import HEADS, build_loss
from .anchor_head_3d import AnchorHead3D, print_tensor
from mmdet.core.bbox.clip_nn import clip_boxes_to_image
# from mmcv.ops import deform_conv2d
from dcn import _DeformConv as deform_conv3d
import ipdb

EPS = 1e-12

class TaskDecomposition3D(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        """
        feat_channels = 256 * stacked_convs
        """
        super(TaskDecomposition3D, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.la_conv1 = nn.Conv3d( self.in_channels,  self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv3d( self.in_channels // la_down_rate,  self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.norm_cfg is None)

    def init_weights(self):
        normal_init(self.la_conv1, std=0.001)
        normal_init(self.la_conv2, std=0.001)
        self.la_conv2.bias.data.zero_()
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w, d = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))
        weight = self.relu(self.la_conv1(avg_feat)) # b, rc, 1, 1, 1
        weight = self.sigmoid(self.la_conv2(weight)) # b, 6, 1, 1, 1

        # here we first compute the product between layer attention weight and conv weight,
        # and then compute the convolution between new conv weight and feature map,
        # in order to save memory and FLOPs. TODO??
        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                          self.reduction_conv.conv.weight.reshape(
                                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w * d)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w, d)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat


@HEADS.register_module()
class TOODHead3D(AnchorHead3D):
    """TOOD: Task-aligned One-stage Object Detection.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    todo: list link of the paper.
    """

    def __init__(self,
                 num_classes,
                 in_channels, 
                 stacked_convs=4,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 num_dcn_on_head=0,
                 anchor_type='anchor_free',
                 initial_loss_cls=dict(
                     type='TaskAlignedLoess',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_dcn_on_head = num_dcn_on_head
        self.anchor_type = anchor_type
        self.epoch = 0 # which would be update in head hook!
        super(TOODHead3D, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner : ATSSAssigner3D = build_assigner(self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.assigner : TaskAlignedAssigner3D = build_assigner(self.train_cfg.assigner)
            # self.alpha = self.train_cfg.alpha
            # self.beta = self.train_cfg.beta
            # SSD sampling=False so use PseudoSampler
            # sampler_cfg = dict(type='PseudoSampler')
            # self.sampler = build_sampler(sampler_cfg, context=self)

        print(f'[TOODHead] spatial dim {self.spatial_dim} anchor/pixle {self.num_anchors}' )
        print(f'[TOODHead] inchannels {self.in_channels} start level {self.start_level}')
        print(f'[TOODHead] bbox scaler by level {self.scales }  FP16{self.fp16_enabled} ')
        

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn_on_head:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = TaskDecomposition3D(self.feat_channels, self.stacked_convs, 
                                                self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition3D(self.feat_channels, self.stacked_convs, 
                                                self.stacked_convs * 8, self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv3d(self.feat_channels, self.num_anchors * self.cls_out_channels, 3, padding=1)
        self.tood_reg = nn.Conv3d(self.feat_channels, self.num_anchors * 6, 3, padding=1)

        self.cls_prob_conv1 = nn.Conv3d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
        self.cls_prob_conv2 = nn.Conv3d(self.feat_channels // 4, 1, 3, padding=1)
        self.reg_offset_conv1 = nn.Conv3d(self.feat_channels * self.stacked_convs, self.feat_channels // 4, 1)
        self.reg_offset_conv2 = nn.Conv3d(self.feat_channels // 4, self.num_anchors * 6 * 3, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])

        self.dcn_bias = torch.zeros((self.num_anchors * 6, ))

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

        normal_init(self.cls_prob_conv1, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_prob_conv2, std=0.01, bias=bias_cls)
        normal_init(self.reg_offset_conv1, std=0.001)
        normal_init(self.reg_offset_conv2, std=0.001)
        self.reg_offset_conv2.bias.data.zero_()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 5D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 5D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 5D-tensor, the channels number is
                    num_anchors * 6.
        """
        num_imgs = len(feats[0])
        featmap_sizes = [featmap.size()[2:] for featmap in feats[self.start_level:]]
        device = feats[0].device
        anchor_list = self.get_anchor_list(
            featmap_sizes, num_imgs, device=device)
        # mini-batch of multi-level anchors to all sample anchors for each level
        level_anchor_list = [torch.cat([anchor_list[i][j] for i in range(len(anchor_list))])
                                     for j in range(len(anchor_list[0]))]

        return multi_apply(self.forward_single, feats[self.start_level:], self.scales, 
                                level_anchor_list, self.anchor_generator.strides)


    def forward_single(self, x, scale, anchor, stride):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            anchor (Tensor): Anchors of a single scale level.
            stride (tuple[Tensor]): Stride of the current scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        b, c, h, w, d = x.shape

        # extract task interactive features; similar to densenet??
        inter_feats = []
        for i, inter_conv in enumerate(self.inter_convs):
            x = inter_conv(x)
            inter_feats.append(x)
        feat = torch.cat(inter_feats, 1) # feat_channels=256 * stacked_convs=6,

        # print_tensor('\n[ToodHead] input feat', x)
        # print_tensor('[ToodHead] interative features', feat)
        # task decomposition with torch.cuda.amp.autocast(enabled=True):
        avg_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))
        cls_feat = self.cls_decomp(feat, avg_feat)
        reg_feat = self.reg_decomp(feat, avg_feat)

        # cls prediction and alignment
        cls_logits = self.tood_cls(cls_feat)
        cls_prob = F.relu(self.cls_prob_conv1(feat))
        cls_prob = self.cls_prob_conv2(cls_prob)
        with torch.cuda.amp.autocast(enabled = False):
            cls_score = torch.clamp(cls_logits.float().sigmoid() * 
                                    cls_prob.float().sigmoid(), min=1e-8).sqrt()

        # reg prediction and alignment
        if self.anchor_type == 'anchor_free':
            reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
            reg_dist = reg_dist.permute(0, 2, 3, 4, 1).reshape(-1, 6)
            reg_bbox = distance2bbox3d(self.anchor_center(anchor) / stride[0], reg_dist
                                        ).reshape(b, h, w, d, 6).permute(0, 4, 1, 2, 3)  # (b, c, h, w, d)
        elif self.anchor_type == 'anchor_based':
            reg_dist = scale(self.tood_reg(reg_feat)).float()
            # print_tensor(f'[THSF] anchor', anchor)
            # print_tensor(f'[THSF] reg dist', reg_dist)
            reg_dist = reg_dist.permute(0, 2, 3, 4, 1).reshape(-1, 6)
            # ipdb.set_trace()
            reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape( # NOTE: divided by stride
                                b, h, w, d, 6 * self.num_anchors).permute(0, 4, 1, 2, 3) / stride[0]
        else:
            raise NotImplementedError
        reg_offset = F.relu(self.reg_offset_conv1(feat))
        reg_offset = self.reg_offset_conv2(reg_offset)
        bbox_pred = self.deform_sampling(reg_bbox.contiguous(), reg_offset.contiguous())
        # print_tensor('bbox pred', bbox_pred)
        # ipdb.set_trace()
        return cls_score, bbox_pred

    def get_anchor_list(self, featmap_sizes, num_imgs, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            num_imgs (int): the number of images in a batch
            device (torch.device | str): Device for returned tensors

        Returns:
            anchor_list (list[Tensor]): Anchors of each image.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        return anchor_list

    def deform_sampling(self, feat, offset): 
        """ Sampling the feature x according to offset.

        Args:
            feat (Tensor): reg bbox Feature, b,num_anchor*6,h,w,d
            offset (Tensor): Spatial offset for for feature sampliing, b,num_anchor*6*3,h,w,d

        args for deform_conv3d: (input, offset, weight, bias = False, 
                                stride=1, padding=0, dilation=1, groups=1, 
                                deformable_groups=1, im2col_step=32)

        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
            offset.shape[1]
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w, d = feat.shape
        weight = feat.new_ones(c, 1, 1, 1, 1).float() # c = num_anchor*6
        if self.dcn_bias.device != feat.device: 
            self.dcn_bias = self.dcn_bias.to(feat.device).float()
        # ipdb.set_trace()
        with torch.cuda.amp.autocast(enabled = False):
            y = deform_conv3d(feat.float(), offset.float(), weight, 
                              self.dcn_bias, 1, 0, 1, c, c, 32)
        return y

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

    def loss_single(self, anchors, cls_score, bbox_pred, labels,
                    label_weights, bbox_targets, alignment_metrics, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1] == stride[2], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 6)
        cls_score = cls_score.permute(0, 2, 3, 4, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, 6)
        bbox_targets = bbox_targets.reshape(-1, 6)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((labels >= 0) & (labels < bg_class_ind)).squeeze(1)

        # count_mask = label_weights > 0
        # fg_mask = labels < bg_class_ind
        # count_anchors = anchors[count_mask]
        # print_tensor('[ToodLoss] anchors', anchors)
        # print_tensor('[ToodLoss] cls_score', cls_score)
        # print_tensor('[ToodLoss] bbox_pred', bbox_pred)
        # print_tensor('[ToodLoss] bbox_targets', bbox_targets)
        # print_tensor('[ToodLoss] labels', labels)
        # print_tensor('[ToodLoss] label weights', label_weights)
        # ipdb.set_trace()

        # classification loss   
        if self.epoch < self.initial_epoch:
            loss_cls = self.initial_loss_cls(
                cls_score, labels, label_weights, avg_factor=1.0)
        else:
            alignment_metrics = alignment_metrics.reshape(-1)
            loss_cls = self.loss_cls(
                cls_score, labels, alignment_metrics, avg_factor=1.0)  # num_total_samples)

        # num_pos_sample = len(pos_inds) 
        # print(f'[THSL] stride {stride[0]} pos {len(pos_inds)} cls loss {loss_cls} align {alignment_metrics.sum()}')
        # print_tensor(f'[THSL] cls prob', cls_score[pos_inds])
        # print_tensor(f'[THSL] alignment metric', alignment_metrics)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0] # NOTE: GT divided by stride

            # print_tensor('bbox pred decode', pos_decode_bbox_pred)
            # print_tensor('bbox gt decode', pos_decode_bbox_targets)
            # regression loss
            if self.epoch < self.initial_epoch:
                pos_bbox_weight = self.centerness_target(
                        pos_anchors, pos_bbox_targets)
            else:
                pos_bbox_weight = alignment_metrics[pos_inds]
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = torch.tensor(0, device = bbox_pred.device)

        return loss_cls, loss_bbox, alignment_metrics.sum(), pos_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors( 
            featmap_sizes, img_metas, device=device) # outer list by image; inner list by level
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            cls_scores,
            bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, alignment_metrics_list) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        cls_avg_factor = sum(cls_avg_factors)
        cls_avg_factor = reduce_mean(cls_avg_factor).item()
        if cls_avg_factor < EPS:
            cls_avg_factor = 1
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        # print(f'[THloss] cls avg {cls_avg_factor}  loss {losses_cls}')

        bbox_avg_factor = sum(bbox_avg_factors)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        if bbox_avg_factor < EPS:
            bbox_avg_factor = 1
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox)

    def centerness_target_2d(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        # for bbox-based
        # gts = self.bbox_coder.decode(anchors, bbox_targets)
        # for point-based
        gts = bbox_targets
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

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
        # if verbose:
        #     print_tensor('[CerterNess] anchors', anchors)
        #     print_tensor('[CerterNess] bbox targets', bbox_targets)
        #     print_tensor('[CerterNess] bbox gt', gts)
        #     print_tensor('[CenterNess] map', centerness)
        #     # pdb.set_trace()
        
        assert not torch.isnan(centerness).any()
        return centerness


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
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
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                img_shape, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W, D).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 6, H, W, D).
            # mlvl_anchors (list[Tensor]): Box reference for a single scale level
            #     with shape (num_total_anchors, 6).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, depth).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, d_scale, w_scale, h_scale, d_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 7), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, tl_z, br_x, br_y, br_z) and the 7-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride in zip(
                cls_scores, bbox_preds, self.anchor_generator.strides):
            assert cls_score.size()[2:] == bbox_pred.size()[2:]
            assert stride[0] == stride[1] == stride[2]

            scores = cls_score.permute(1, 2, 3, 0).reshape(-1, self.cls_out_channels)
            bbox_pred = bbox_pred.permute(1, 2, 3, 0).reshape(-1, 6) * stride[0]
            # print_tensor(f'[Tood] test stride {stride}', scores)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            
            # NOTE: limiting the bbox boundary may not be necessary
            # bboxes = clip_boxes_to_image(bbox_pred, img_shape = img_shape)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_3d(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for TOOD head.

        This method is almost the same as `AnchorHead3D.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        all_cls_scores = torch.cat(
            [cls_score.permute(0, 2, 3, 4, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in
             cls_scores], 1)
        all_bbox_preds = torch.cat(
            [bbox_pred.permute(0, 2, 3, 4, 1).reshape(num_imgs, -1, 6) * stride[0] # NOTE: times stride
                for bbox_pred, stride in zip(bbox_preds, self.anchor_generator.strides)],
            dim = 1)

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # anchor_list: list(b * [-1, 4])
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, pos_assigned_gt_inds_list,
         assign_metrics_list, assign_ious_list, inside_flags_list) = multi_apply(
             self._get_target_single,
             all_cls_scores,
             all_bbox_preds,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
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

        if self.epoch < self.initial_epoch:
            norm_alignment_metrics_list = [bbox_weights[:, :, 0] for bbox_weights in bbox_weights_list]
            # ipdb.set_trace()
        else:
            # for alignment metric
            all_norm_alignment_metrics = []
            for i in range(num_imgs):
                inside_flags = inside_flags_list[i]
                image_norm_alignment_metrics = all_label_weights[i].new_zeros(all_label_weights[i].shape[0])
                image_norm_alignment_metrics_inside = all_label_weights[i].new_zeros(inside_flags.long().sum())
                pos_assigned_gt_inds = pos_assigned_gt_inds_list[i]
                pos_inds = pos_inds_list[i]
                class_assigned_gt_inds = torch.unique(pos_assigned_gt_inds)
                for gt_inds in class_assigned_gt_inds:
                    gt_class_inds = pos_inds[pos_assigned_gt_inds == gt_inds]
                    pos_alignment_metrics = assign_metrics_list[i][gt_class_inds]
                    pos_ious = assign_ious_list[i][gt_class_inds]
                    pos_norm_alignment_metrics = pos_alignment_metrics / (pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
                    image_norm_alignment_metrics_inside[gt_class_inds] = pos_norm_alignment_metrics

                image_norm_alignment_metrics[inside_flags] = image_norm_alignment_metrics_inside
                all_norm_alignment_metrics.append(image_norm_alignment_metrics)

            norm_alignment_metrics_list = images_to_levels(all_norm_alignment_metrics,
                                                  num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, norm_alignment_metrics_list)


    def _get_target_single(self,
                           cls_scores,
                           bbox_preds,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 6)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 6).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 6).
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
                pos_inds (Tensor): Indices of postive anchor with shape
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
        cls_scores = cls_scores[inside_flags, :]
        bbox_preds = bbox_preds[inside_flags, :]
        

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        if self.epoch < self.initial_epoch:
            assign_result = self.initial_assigner.assign(anchors, num_level_anchors_inside,
                                                 gt_bboxes, gt_bboxes_ignore,
                                                 gt_labels)
            assign_ious = assign_result.max_overlaps
            assign_metrics = None
        else:
            assign_result = self.assigner.assign(cls_scores, bbox_preds,
                                             anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
            assign_ious = assign_result.max_overlaps
            assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes, cls_scores = cls_scores)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # ipdb.set_trace()
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

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
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds, assign_metrics, assign_ious, inside_flags)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
