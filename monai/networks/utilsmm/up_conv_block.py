import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer

class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convoluton in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        super(UpConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out


# import torch.nn.functional as F
# from mmcv.cnn.utils import xavier_init
# from mmcv.cnn import UPSAMPLE_LAYERS
# @UPSAMPLE_LAYERS.register_module(name='pixel_shuffle3d')
# class PixelShufflePack3D(nn.Module):
#     """Pixel Shuffle upsample layer.

#     This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
#     achieve a simple upsampling with pixel shuffle.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         scale_factor (int): Upsample ratio.
#         upsample_kernel (int): Kernel size of the conv layer to expand the
#             channels.
#     """

#     def __init__(self, in_channels, out_channels, scale_factor,
#                  upsample_kernel):
#         super(PixelShufflePack3D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.scale_factor = scale_factor
#         self.upsample_kernel = upsample_kernel
#         self.upsample_conv = nn.Conv2d(
#             self.in_channels,
#             self.out_channels * scale_factor * scale_factor,
#             self.upsample_kernel,
#             padding=(self.upsample_kernel - 1) // 2)
#         self.init_weights()

#     def init_weights(self):
#         xavier_init(self.upsample_conv, distribution='uniform')

#     def forward(self, x):
#         x = self.upsample_conv(x)
#         x = F.pixel_shuffle(x, self.scale_factor)
#         return x