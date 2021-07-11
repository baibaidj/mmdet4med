"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Norm

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self,
                 dim=3,
                 n_channels=1,
                 start_filts=18,
                 end_filts=36,
                 n_latent_dims=0,
                 act=Act.PRELU,
                 norm=Norm.INSTANCE,
                 res_architecture='resnet50',
                 sixth_pooling=False,
                 operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param act: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts = start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = sixth_pooling
        self.dim = dim
        self.act = act
        self.strides3d = (2, 2, 2)  # original_param = (2, 2, 1)
        if operate_stride1:
            self.C0 = nn.Sequential(Convolution(self.dim, n_channels, start_filts, kernel_size=3, padding=1, norm=norm, act=act),
                                    Convolution(self.dim, start_filts, start_filts, kernel_size=3, padding=1, norm=norm, act=act))

            self.C1 = Convolution(self.dim,start_filts, start_filts, kernel_size=7, strides=self.strides3d if self.dim == 3 else 2, padding=3, norm=norm, act=act)

        else:
            self.C1 = Convolution(self.dim,n_channels, start_filts, kernel_size=7, strides=self.strides3d if self.dim == 3 else 2, padding=3, norm=norm, act=act)

        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if self.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=self.strides3d, padding=1))
        C2_layers.append(self.block(self.dim, start_filts, start_filts, stride=1, norm=norm, act=act,
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(self.dim, start_filts_exp, start_filts, norm=norm, act=act))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(self.dim, start_filts_exp, start_filts * 2, stride=2, norm=norm, act=act,
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(self.dim, start_filts_exp * 2, start_filts * 2, norm=norm, act=act))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(
            self.dim, start_filts_exp * 2, start_filts * 4, stride=2, norm=norm, act=act, downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(self.dim, start_filts_exp * 4, start_filts * 4, norm=norm, act=act))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(
            self.dim, start_filts_exp * 4, start_filts * 8, stride=2, norm=norm, act=act, downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(self.dim, start_filts_exp * 8, start_filts * 8, norm=norm, act=act))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(
                self.dim, start_filts_exp * 8, start_filts * 16, stride=2, norm=norm, act=act, downsample=(start_filts_exp * 8, 2, 2)))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(self.dim, start_filts_exp * 16, start_filts * 16, norm=norm, act=act))
            self.C6 = nn.Sequential(*C6_layers)

        if self.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 2), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 2), mode='trilinear')

        self.out_channels = end_filts
        self.P5_conv1 = Convolution(self.dim, start_filts*32 + n_latent_dims, self.out_channels, kernel_size=1, strides=1, act=None) #
        self.P4_conv1 = Convolution(self.dim, start_filts*16, self.out_channels, kernel_size=1, strides=1, act=None)
        self.P3_conv1 = Convolution(self.dim, start_filts*8, self.out_channels, kernel_size=1, strides=1, act=None)
        self.P2_conv1 = Convolution(self.dim, start_filts*4, self.out_channels, kernel_size=1, strides=1, act=None)
        self.P1_conv1 = Convolution(self.dim, start_filts, self.out_channels, kernel_size=1, strides=1, act=None)

        if operate_stride1:
            self.P0_conv1 = Convolution(self.dim, start_filts, self.out_channels, kernel_size=1, strides=1, act=None)
            self.P0_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)

        self.P1_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)
        self.P2_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)
        self.P3_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)
        self.P4_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)
        self.P5_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)

        if self.sixth_pooling:
            self.P6_conv1 = Convolution(self.dim, start_filts * 64, self.out_channels, kernel_size=1, strides=1, act=None)
            self.P6_conv2 = Convolution(self.dim, self.out_channels, self.out_channels, kernel_size=3, strides=1, padding=1, act=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list



class ResBlock(nn.Module):

    def __init__(self, dim, start_filts, planes, stride=1, downsample=None, norm=None, act='act'):
        super(ResBlock, self).__init__()
        self.conv1 = Convolution(dim, start_filts, planes, kernel_size=1, strides=stride, norm=norm, act=act)
        self.conv2 = Convolution(dim, planes, planes, kernel_size=3, padding=1, norm=norm, act=act)
        self.conv3 = Convolution(dim, planes, planes * 4, kernel_size=1, norm=norm, act=None)
        self.act = nn.ReLU(inplace=True) if act == 'act' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = Convolution(dim, downsample[0], downsample[0] * downsample[1], kernel_size=1, strides=downsample[2], norm=norm, act=None)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


if __name__ == '__main__':
    tensor = torch.rand([1, 1, 128, 128, 128], dtype=torch.float)
    fpn = FPN()
    out = fpn(tensor)
    print(out[0].size(), out[1].size(), out[2].size(), out[3].size())

