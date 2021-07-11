import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from scipy import ndimage as ndi
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)

def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (Tensor): The grid to be denormalize, range [0, 1].
    Returns:
        Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0




def point_sample_3d(input, points, align_corners=False, **kwargs):
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (Tensor): Feature map, shape (N, C, D, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 3) or (N, Dgrid, Hgrid, Wgrid, 3).
        align_corners (bool): Whether align_corners. Default: False

    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Dgrid, Hgrid, Wgrid).
    """
    assert points.dim() in [3, 5]
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        N, P, d = points.shape
        points = points.view(N, P, 1, 1, d)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        N, C, *_ = output.shape
        output = output.view(N, C, -1)
    return output

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


class SemanticFlowUpsample(nn.Module):
    """

    this method is a better way to integrate the high-level low resolution features with the low-level high resolution features
    
    adopted from https://github.com/lxtGH/SFSegNets
    
    """

    def __init__(self, inplane, outplane, kernel_size=3, conv_cfg = None):
        super(SemanticFlowUpsample, self).__init__()
        
        self.conv_cfg = conv_cfg
        self.is_3d = False if self.conv_cfg is None else (True if '3d' in self.conv_cfg.get('type', '').lower() else False)

        self.down_h = build_conv_layer(conv_cfg, inplane, outplane, 1, bias=False)
        self.down_l = build_conv_layer(conv_cfg, inplane, outplane, 1, bias=False)
        self.flow_make = build_conv_layer(conv_cfg, outplane*2, 3 if self.is_3d else 2, 
                        kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output