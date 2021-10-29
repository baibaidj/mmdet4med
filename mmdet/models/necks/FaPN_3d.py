import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer
from mmcv.runner import BaseModule
from typing import Sequence, List
from ..builder import NECKS
from ..utils.ccnet_pure import print_tensor
from dcn import DeformConv as DeformConv3D
import pdb, torch


@NECKS.register_module()
class FaPN3D(BaseModule):
    r"""Feature Pyramid Network.

    modified based on UFPN from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/arch/decoder/base.py

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 fixed_out_channels,
                 num_outs,
                 start_level=2,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 min_out_channels = 8, 
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None, verbose = False, 
                 dcn_cfg = dict(kernel_size = 3, deform_groups = 1), 
                 upsample_cfg=dict(type='deconv3d', kernel_size = (2,2,2), stride = (2,2,2) ),
                 init_cfg=dict(
                     type='Xavier', layer='Conv3d', distribution='uniform'), 
                 is_double_chn = True,     
                ):
        super(FaPN3D, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.fixed_out_channels = fixed_out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        # self.upsample_mode = 'trilinear'
        self.deconv_cfg = upsample_cfg.copy()
        self.min_out_channels = min_out_channels
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.out_channels = self.compute_output_channels(is_double_chn)
        self.up_ops = self.build_upsample_layers(conv_cfg, dcn_cfg = dcn_cfg)
        print(f'[FaPN3D] input channels {self.in_channels} out channels {self.out_channels}')
        # print(f'[FaPN3D] up ops\n', self.up_ops)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.verbose = verbose
        for i in range(0, self.backbone_end_level):
            l_conv = FeatureSelectionModule(
                in_channels[i],
                self.out_channels[i],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.out_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = fixed_out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    fixed_out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def compute_output_channels(self, is_double_chn = True) -> List[int]:
        """
        Compute number of output channels
        for upper most two levels, the channels can stay the same as the encoder featmap. 
        Returns:
            List[int]: number of output channels for each level
        """
        out_channels = [self.fixed_out_channels] * self.num_ins

        if isinstance(self.start_level, int) and self.start_level > 0 : #2345
            ouput_levels = list(range(self.num_ins)) # encoder outputing levels, 01234
            # filter for levels above decoder levels
            ouput_levels = [ol for ol in ouput_levels if ol < self.start_level]
            assert max(ouput_levels) < self.start_level, "Can not decrease channels below decoder level"
            for ol in ouput_levels[::-1]: # 1, 0 
                oc = max(self.min_out_channels, self.in_channels[ol]* (2 if is_double_chn else 1 ))
                out_channels[ol] = oc
        return out_channels

    def build_upsample_layers(self, conv_cfg, norm_cfg = None, act_cfg = None, 
                              dcn_cfg = dict()):
        up_ops = nn.ModuleList()
        for i in range(0, self.backbone_end_level):
            if i == 0:
                up = None #nn.Identity()
            elif (i > 0) and (i <= self.start_level):
                # up = nn.Upsample(scale_factor=2, mode= self.upsample_mode, align_corners=True)
                # if not (self.out_channels[i] == self.out_channels[i - 1]):
                #     _conv = ConvModule(self.out_channels[i],
                #                         self.out_channels[i - 1], 
                #                         1, conv_cfg=conv_cfg, 
                #                         norm_cfg = None, 
                #                         act_cfg=None)
                #     up = nn.Sequential(up, _conv)
                up = nn.Sequential(build_upsample_layer(
                                                cfg=self.deconv_cfg,
                                                in_channels=self.out_channels[i],
                                                out_channels=self.out_channels[i-1],
                                                bias = False),
                                                # build_norm_layer(norm_cfg, up_channels)[1],
                                                # nn.ReLU(inplace=True)
                                            )
            else:
                up = FeatureAlign_V2(self.out_channels[i], 
                                    self.out_channels[i - 1], 
                                    dcn_kernel= dcn_cfg.get('kernel_size', 3), 
                                    deform_groups= dcn_cfg.get('deform_groups', 1), 
                                    conv_cfg = conv_cfg, 
                                    norm_cfg = norm_cfg, 
                                    act_cfg = act_cfg
                                    )
            up_ops.append(up)
        return up_ops

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # pdb.set_trace()
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1): # 5,4,3,2,1
            # print_tensor(f'[fpn3d ups] {i} {self.up_ops[i]}', laterals[i])

            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if i <= self.start_level:
                up_feat = self.up_ops[i](laterals[i])
            else:
                up_feat = self.up_ops[i](laterals[i], laterals[i - 1])
            # print(f'\n {self.up_ops[i]}')
            # print_tensor(f'[FaPN] ups {i} ', laterals[i])
            # print_tensor(f'[FaPN] ups {i - 1}', laterals[i- 1])
            laterals[i - 1] = laterals[i - 1] + up_feat

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # bb = [print_tensor(f'[FPNeck] inputlevel {i}', o) for i, o in enumerate(inputs)]
        # level_hasnan = [torch.isnan(a).any() for i, a in enumerate(outs)]
        if self.verbose: 
            aa = [print_tensor(f'[FPNeck] level {i}', o) for i, o in enumerate(outs)] #hasnan {level_hasnan[i]}
        # if any(level_hasnan): pdb.set_trace()
        return tuple(outs)



class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size = 1, 
                conv_cfg = dict(type = 'Conv3d'), norm_cfg = None, act_cfg = None,
                inplace = False):
        super(FeatureSelectionModule, self).__init__()

        self.conv_atten = ConvModule(in_chan, 
                                    in_chan, 
                                    kernel_size, 
                                    conv_cfg=conv_cfg, 
                                    norm_cfg = None, 
                                    act_cfg = act_cfg, 
                                    bias=False,)
        self.sigmoid = nn.Sigmoid()
        self.conv =  ConvModule(in_chan, 
                                out_chan, 
                                kernel_size, 
                                conv_cfg=conv_cfg, 
                                norm_cfg = norm_cfg, 
                                act_cfg = act_cfg, 
                                bias=False)
        if isinstance(conv_cfg, dict) and '3d' in conv_cfg.get('type', ''):
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool  = nn.AdaptiveAvgPool3d((1, 1))

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten((self.avg_pool(x))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


kernel2pad = lambda a : (a - 1)//2

class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, channels_src=128, 
                channels_dst=128, 
                dcn_kernel = 3, 
                deform_groups = 8, 
                conv_cfg = dict(type = 'Conv3d'), 
                norm_cfg = None, 
                act_cfg = None):
        super(FeatureAlign_V2, self).__init__()
        # self.is_3d = isinstance(conv_cfg, dict) and ('3d' in conv_cfg.get('type', ''))
        # self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.offset = ConvModule(channels_dst + channels_src, 
                                 dcn_kernel**3 * 3 * deform_groups, 
                                1, 
                                conv_cfg=conv_cfg, 
                                norm_cfg = norm_cfg, 
                                act_cfg = act_cfg, 
                                bias=False,)

        # self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True)
        self.dcpack_L2 = DeformConv3D( # NOTE: 
                            channels_src,
                            channels_dst,
                            dcn_kernel,
                            padding = kernel2pad(dcn_kernel),
                            deform_groups=deform_groups)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_1x_src, feat_2x_dst):
        """
        

        Args:
            feat_2x_src ([type]): high resolution, low level feature map, Pi
            feat_1x_dst ([type]): low resolution, high level feature map, Pi+1
            main_path ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        HWD = feat_2x_dst.size()[2:]
        if feat_2x_dst.size()[2:] != feat_1x_src.size()[2:]:
            feat_up = F.interpolate(feat_1x_src, HWD, mode='trilinear', align_corners=False)
        else:
            feat_up = feat_1x_src
        # feat_arm = self.lateral_conv(feat_2x_low)  # 0~1 * feats

        offset = self.offset(torch.cat([feat_2x_dst, feat_up * 2], dim=1))  # concat for offset by compute the dif

        # print_tensor('[FAM] 1x src', feat_1x_src)
        # print_tensor('[FAM] 2x dst', feat_2x_dst )
        with torch.cuda.amp.autocast(enabled = False):
            # print_tensor('[FAM] 1x src upsample', feat_up )
            # print_tensor('[FAM] offset', offset)
            dcn_result = self.dcpack_L2(feat_up.float(), offset.float())

        feat_align = self.relu(dcn_result) 
        return feat_align
