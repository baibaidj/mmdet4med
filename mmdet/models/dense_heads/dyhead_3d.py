
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from ..utils.dyrelu import torch, nn, F, DYReLU, h_sigmoid
from ..utils.modulated_deform_conv import ModulatedDeformConv3d

from .atss_head_3d_noc import ATSSHead3DNOC, HEADS, multi_apply, print_tensor
import ipdb


@HEADS.register_module()
class DynamicHead3D(ATSSHead3DNOC):

    def __init__(self,
                 *arg, 
                 kernel_size = 3, 
                 num_dytower = 6, 
                 dy_init_cfg = dict(type='Normal', layer='Conv3d', std=0.01),
                 **kwargs):
        self.kernel_size = kernel_size
        self.num_dytower = num_dytower
        self.dy_init_cfg = dy_init_cfg
        print(f'[DYHead] kernelsize {kernel_size} num_dytower {num_dytower}')
        super(DynamicHead3D, self).__init__(*arg, **kwargs)

        self._init_layers_dy() 

        # print('[DYHead] start level', self.start_level)

    def _init_layers_dy(self):
        """Initialize layers of the head."""
        dyhead_tower = []
        for i in range(self.num_dytower):
            dy_block_i = DyConv3D(self.in_channels if i == 0 else self.feat_channels, 
                                  self.feat_channels, 
                                  kernel_size = self.kernel_size, 
                                  conv_cfg = self.conv_cfg, 
                                  norm_cfg=self.norm_cfg, 
                                  start_level=self.start_level, 
                                  init_cfg=self.dy_init_cfg
                                  )
            dyhead_tower.append(dy_block_i)

        self.dyhead_tower = nn.Sequential(*dyhead_tower)


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
        out_feats = self.dyhead_tower(feats)
        return multi_apply(self.forward_single, out_feats[self.start_level:], self.scales)


class DCNv2_Norm_3D(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, init_cfg = None):
        super(DCNv2_Norm_3D, self).__init__(init_cfg)

        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = ModulatedDeformConv3d(in_channels, out_channels, 
                                            kernel_size = kernel_size, stride=stride, 
                                            padding = (kernel_size-1)//2, 
                                            # in_step=128
                                            )
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input,  offset, mask):
        # offset, mask = offset.contiguous(), mask.contiguous()
        if self.stride == 1 and (offset.shape[2:] != input.shape[2:]):
            # print('Do resize for offset and mask')
            offset = F.interpolate(offset, size =  input.shape[2:], mode = 'trilinear', align_corners= False)
            mask = F.interpolate(mask, size =  input.shape[2:], mode = 'trilinear', align_corners= False)

        with torch.cuda.amp.autocast(enabled = False):
            # print(f'input {input.shape} offset {offset.shape} mask {mask.shape}')
            # ipdb.set_trace()
            x = self.conv(input.contiguous().float(), offset.float(), mask.float())
        x = self.bn(x)
        return x

class DyConv3D(BaseModule):
    def __init__(self, in_channels=256, out_channels=256, 
                kernel_size = 3, 
                conv_cfg = dict(type = 'DCN3dv2'),
                norm_cfg = dict(type = 'GN', num_groups = 8), 
                start_level = 0, 
                init_cfg = None):
        super(DyConv3D, self).__init__(init_cfg)
        """Initialize layers of the head."""
        # self.DyConv = nn.ModuleList()
        # self.DyConv_upper =  ConvModule(in_channels, 
        #                                 out_channels, 
        #                                 kernel_size, 
        #                                 stride = 1, 
        #                                 conv_cfg = conv_cfg, 
        #                                 norm_cfg = norm_cfg)
        self.in_channels = in_channels
        self.feat_channels = out_channels
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.start_level = start_level
        self.DyConv_upper = DCNv2_Norm_3D(self.in_channels, self.feat_channels, 
                            kernel_size = self.kernel_size, stride = 1)
            # build_norm_layer(self.norm_cfg, self.feat_channels)[1]

        self.DyConv_middle = DCNv2_Norm_3D(self.in_channels, self.feat_channels, 
                            kernel_size = self.kernel_size, stride = 1)
            # build_norm_layer(self.norm_cfg, self.feat_channels)[1]

        self.DyConv_lower = DCNv2_Norm_3D(self.in_channels, self.feat_channels, 
                            kernel_size = self.kernel_size, stride = 2)
            # build_norm_layer(self.norm_cfg, self.feat_channels)[1]

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(self.in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(self.in_channels, self.feat_channels, 
                            conv_cfg = self.conv_cfg, norm_cfg = self.norm_cfg)

        self.offset_channel = self.kernel_size**3
        self.offset = nn.Conv3d(self.in_channels, 
                                self.offset_channel * 4, 
                                kernel_size=self.kernel_size, stride=1, 
                                padding=(self.kernel_size-1)//2)

    #     self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.normal_(m.weight.data, 0, 0.01)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()

    def forward(self, feats): # "res3", "res4", "res5"
        num_levels = len(feats)
        out_feats = [feats[i] for i in range(self.start_level)]
        for level in range(self.start_level, num_levels): # 1/4, 1/8, 1/16, 1/32
            feature = feats[level]
            offset_mask = self.offset(feature)

            # print_tensor(f'\n[Feat] lvl {level} feature', feature)
            offset = offset_mask[:, :self.offset_channel * 3, ...].contiguous() # B,18,H,W() # 3x3x3 x3 , coordinates for each kernel element
            mask = offset_mask[:, self.offset_channel*3:, ...].contiguous().sigmoid() # B,9,H,W (0~1), # 3x3 x 1, weight to rescale the value offset. 
            conv_args = dict(offset=offset, mask=mask)
            # ipdb.set_trace()
            # print_tensor(f'[Feat] lvl {level} offset', offset)
            # print_tensor(f'[Feat] lvl {level} mask', mask)

            temp_fea = [self.DyConv_middle(feature, **conv_args)] # stride = 1
            if level > self.start_level:
                # print_tensor(f'[feat] level - 1 {(level - 1)}', feats[level - 1])
                from_lower = self.DyConv_lower(feats[level - 1], **conv_args)
                temp_fea.append(from_lower) # stride = 2, downsample
            if level < num_levels - 1:
                from_upper = self.DyConv_upper(feats[level + 1], **conv_args)
                from_upper = F.interpolate(from_upper, size=feature.shape[2:], mode = 'trilinear', align_corners=False)
                temp_fea.append(from_upper)
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            feat_this_level = self.relu(mean_fea)
            out_feats.append(feat_this_level)

        return out_feats