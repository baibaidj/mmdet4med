import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head_med import *
from ..utils.implicit_semantic_data_aug import ISDALoss


@HEADS.register_module()
class FcnMetricHead3D(BaseDecodeHeadMed):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        is_use_isda (boo): if use implicit semantic data augmentation
        isda_lambda (float) : 'The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=False,
                 is_use_isda = False, 
                 isda_lambda = 2.5,
                 start_iters = 1,
                 max_iters = 4e5,
                 is_isda_3d = False,
                 use_cls_seg = True,
                 **kwargs):
        assert isinstance(num_convs, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.is_isda_3d = is_isda_3d
        self.use_cls_seg = use_cls_seg
        super(FcnMetricHead3D, self).__init__(**kwargs)
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.is_use_isda = is_use_isda
        self.isda_lambda = isda_lambda
        self._iter = start_iters
        self._max_iters = max_iters
        if is_use_isda:
            self.isda_augmentor = ISDALoss(self.final_channel, self.num_classes, is_3d = self.is_isda_3d)
        if not self.use_cls_seg: self.cls_seg = nn.Identity()

    def forward(self, inputs):
        """Forward function."""
        # ratio = args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        x = self._transform_inputs(inputs)
        feat_map = self.convs(x) if self.num_convs > 0 else x
        if self.concat_input:
            feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
            
        output = self.cls_seg(feat_map)
        if self.is_use_isda and self.training :
            return output, feat_map.detach()
        else:
            return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize_3d(
            input=seg_logit,
            size=self.get_shape(seg_label),
            mode=self.get_mode(),
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        # seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss