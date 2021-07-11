import torch

from ..builder import HEADS
# from .fcn_head import FCNHead
from .fcn_head_3d import FCNHead3D
from ..utils.ccnet_pure import CrissCrossAttention3D, CrissCrossAttention

# from .decode_head_med import BaseDecodeHeadMed, force_fp32, accuracy, resize_3d, print_tensor
# from mmcv.cnn import ConvModule
# from mmcv.runner import force_fp32

@HEADS.register_module()
class CCHeadAug(FCNHead3D):
    """CCNet: Criss-Cross Attention for Semantic Segmentation.

    This head is the implementation of `CCNet
    <https://arxiv.org/abs/1811.11721>`_.

    Args:
        recurrence (int): Number of recurrence of Criss Cross Attention
            module. Default: 2.
    """

    def __init__(self, recurrence=2, **kwargs):
        if CrissCrossAttention is None:
            raise RuntimeError('Please install mmcv-full for '
                               'CrissCrossAttention ops')
        super(CCHeadAug, self).__init__(num_convs=2, **kwargs)
        self.recurrence = recurrence
        self.cca = CrissCrossAttention(self.channels)

    def forward(self, inputs):
        """Forward function."""
        # ratio = args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        x = self._transform_inputs(inputs)
        output0 = self.convs[0](x)
        for _ in range(self.recurrence):
            output0 = self.cca(output0)
        feat_map = self.convs[1](output0)
        if self.concat_input:
            feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
        output = self.cls_seg(feat_map)
        if self.is_use_isda and self.training :
            return output, feat_map.detach()
        else:
            return output

@HEADS.register_module()
class CCHead3D(FCNHead3D):
    """CCNet: Criss-Cross Attention for Semantic Segmentation.

    This head is the implementation of `CCNet
    <https://arxiv.org/abs/1811.11721>`_.

    Args:
        recurrence (int): Number of recurrence of Criss Cross Attention
            module. Default: 2.
    """

    def __init__(self, recurrence=3, **kwargs):
        if CrissCrossAttention is None:
            raise RuntimeError('Please install mmcv-full for '
                               'CrissCrossAttention ops')
        super(CCHead3D, self).__init__(num_convs=2, **kwargs)
        self.recurrence = recurrence
        self.cca = CrissCrossAttention3D(self.channels)


    def forward(self, inputs):
        """Forward function."""
        # ratio = args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        x = self._transform_inputs(inputs)
        output0 = self.convs[0](x)
        for _ in range(self.recurrence):
            output0 = self.cca(output0)
        feat_map = self.convs[1](output0)
        if self.concat_input:
            feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
        output = self.cls_seg(feat_map)
        if self.is_use_isda and self.training :
            return output, feat_map.detach()
        else:
            return output
