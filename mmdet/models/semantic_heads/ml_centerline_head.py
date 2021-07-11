import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from ..builder import HEADS
from .decode_head_med import *
# from ..utils import get_root_logger
# from mmcv.runner import load_checkpoint
# from skimage.morphology import ball


@HEADS.register_module()
class TopoDistanceHead(BaseDecodeHeadMed):
    """
    adopted from TopNet https://link.springer.com/chapter/10.1007/978-3-030-59725-2_2

    This is a metric learning head, which minimize difference between the L2 norm of two pixel feature space and their physical distance
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 radius = 5, 
                 **kwargs):
        assert isinstance(num_convs, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.radius = radius
        super(TopoDistanceHead, self).__init__(**kwargs)
        self.conv_seg = self.conv_final(self.final_channel, self.num_classes, kernel_size=1)
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
        

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x) if self.num_convs > 0 else x
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        
        # print_tensor('DstMapOut %s'% self.num_classes, output)
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

        # seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            # weight=seg_weight,
            ignore_index=self.ignore_index)
        # loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss