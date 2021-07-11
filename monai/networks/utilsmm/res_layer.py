from mmcv.cnn import build_conv_layer, build_norm_layer
from torch import nn as nn


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)



class ResLayer3D(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        multi_grid (int | None): Multi grid dilation rates of last
            stage. Default: None
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 dilation=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        
        self.block = block
        st_per_block = 2 if 'basic' in block.__name__.lower() else 1
        global_ix = kwargs.pop('global_ix', None)
        # print(block, dir(block), block.__name__.lower(), st_per_block)
        # print('res1', global_ix)
        downsample = None
        stride = [stride] * 3 if type(stride) is int else stride
        dilation = [dilation] * 3 if type(dilation) is int else dilation
        assert len(stride) == 3

        if max(stride) != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            # check dilation for dilated ResNet
            if avg_down and (max(stride) != 1 or max(dilation) != 1):
                conv_stride = 1
                downsample.append(
                    nn.AvgPool3d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if max(dilation) > 1 and contract_dilation:
                first_dilation = [max(dilation[i] // 2, 1) for i in dilation] 
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        kwargs1 = dict(inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs)
        if global_ix is not None: kwargs1['global_ix'] = global_ix

        layers.append(block(**kwargs1))

        global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
        # print('res2', global_ix)
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            kwargs_loop = dict(inplanes=inplanes,
                                planes=planes,
                                stride=1,
                                dilation=dilation if multi_grid is None else multi_grid[i],
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                **kwargs)
            if global_ix is not None: kwargs_loop['global_ix'] = global_ix

            layers.append(block(**kwargs_loop))

            global_ix = global_ix + 1 * st_per_block if global_ix is not None else None
            # print('resl', global_ix)
        super(ResLayer3D, self).__init__(*layers)


