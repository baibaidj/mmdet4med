import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head_med import (BaseDecodeHeadMed,  print_tensor, 
                normal_init, force_fp32, 
                accuracy, chn2last_order)

from ...utils.resize import bnchw2bchw, resize_3d
# from mmcv.runner import auto_fp16

@HEADS.register_module()
class FlowHead3D(BaseDecodeHeadMed):
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
                 concat_input = False,
                 verbose = False,
                 acc_gt_index = 0,
                 num_classes = 3, 
                 input_shape = (192, 192, 192), 
                 integrate_steps = 7, 
                 loss_match = None, 
                 loss_reg = None, 
                 **kwargs):
        assert isinstance(num_convs, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.acc_gt_index = acc_gt_index
        super(FlowHead3D, self).__init__(num_classes = num_classes, **kwargs)
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
                
        self.verbose = verbose
        self.transformer = SpatialTransformer(input_shape)
        self.integration = VecInt(input_shape, integrate_steps) if integrate_steps > 0 else None


    # def init_weights(self):
    #     """Initialize weights of classification layer."""

    #     if isinstance(self.conv_seg, torch.nn.ModuleList):
    #         for conv_seg in self.conv_seg:
    #             normal_init(conv_seg, mean=0, std=1e-5)
    #     else:
    #         if isinstance(self.conv_seg, (nn.Conv2d, nn.Conv3d)):
    #             normal_init(self.conv_seg, mean=0, std=1e-5)

    def forward(self, inputs):
        """Forward function."""
        # ratio = args.lambda_0 * global_iteration / args.num_steps # training progress as percentage 
        if self.verbose: 
            for i, ip in enumerate(inputs): print_tensor(f'[FCNHead] input {i}', ip)
        x = self._transform_inputs(inputs)
        feat_map = self.convs(x) if self.num_convs > 0 else x
        if self.concat_input:
            feat_map = self.conv_cat(torch.cat([x, feat_map], dim=1))
            
        flow_field = self.cls_seg(feat_map)
        if self.verbose: 
            print_tensor('[MatchHead] finalfeat', feat_map)
            print_tensor('[MatchHead] fcnout', flow_field)

        return flow_field

    def forward_train(self, inputs, img_metas, targets, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        flow_field = self.forward(inputs)

        moving_image, fixed_image = targets[:, 0], targets[:, 1]
        
        if self.integration is not None:
            flow_field_final = self.integration(flow_field)
        else: flow_field_final = flow_field

        if moving_image.shape != flow_field_final.shape:
            flow_field_final = resize_3d(flow_field_final, size = moving_image.shape, 
                                 mode = 'trilinear', 
                                 align_corners= True, 
                                 warning = False)

        moved_image = self.transformer(moving_image, flow_field_final)

        losses = self.losses(flow_field, moved_image, fixed_image)

        return losses


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, flow_field, moved_image, fixed_image):
        """Compute segmentation loss."""
        loss = dict()
        # print_tensor('LOSS: seg logit', seg_logit)
        # print_tensor('LOSS: seg label', seg_label)
        # seg_label should be B, 1, S, H, W

        loss['loss_match'] = self.loss_match(
            fixed_image, 
            moved_image,
            weight=None)
        loss['loss_reg'] = self.loss_reg(flow_field)

        return loss

    def simple_test(self, feats, img_metas, targets, rescale=False):
        
        flow_field = self.forward(feats)

        moving_image, fixed_image = targets[:, 0], targets[:, 1]
        
        if self.integration is not None:
            flow_field_final = self.integration(flow_field)
        else: flow_field_final = flow_field

        if moving_image.shape != flow_field_final.shape:
            flow_field_final = resize_3d(flow_field_final, size = moving_image.shape, 
                                 mode = 'trilinear', 
                                 align_corners= True, 
                                 warning = False)

        moved_image = self.transformer(moving_image, flow_field_final)

        return flow_field_final, moved_image


import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid.to(src.dtype) + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps = 7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec