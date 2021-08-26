import numpy as np
import torch
from torch.nn.modules.utils import _pair, _triple

from .builder import PRIOR_GENERATORS
from mmdet.utils import print_tensor
import pdb

@PRIOR_GENERATORS.register_module()
class AnchorGenerator3D:
    """Standard anchor generator for 3D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h, d).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0., 
                 verbose = False):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'
        self.verbose = verbose
        # calculate base sizes of anchors
        self.strides = [_triple(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'
        if self.verbose: 
            print('[AnchorGen] basesize', self.base_sizes)
        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()
        if self.verbose: 
            for i, an in enumerate(self.base_anchors):
                print(f'[AnchorGen] base anchor level {i} \n', an) 

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self, verbose = True):
        """Generate base anchors.
        base_sizes: [1, 2, 4, 8, 16]
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            anchor_i = self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center)
            if verbose: print(f'[BaseAnchor] level {i} basesize {base_size} ', anchor_i)
            multi_level_base_anchors.append(anchor_i)
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (torch.Tensor): Scales of the anchor.
            ratios (torch.Tensor): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = h = d = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
            z_center = self.center_offset * d
        else:
            x_center, y_center, z_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = d_ratios = 1 / h_ratios

        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
            ds = (d * d_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)
            ds = (d * scales[:, None] * d_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, z_center - 0.5 * ds, 
            x_center + 0.5 * ws, y_center + 0.5 * hs, z_center + 0.5 * ds
        ]
        base_anchors = torch.stack(base_anchors, dim=-1) # nx6

        return base_anchors

    def _meshgrid(self, x, y, z, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xxx = x[:, None, None].repeat(1, y.shape[0], z.shape[0]).view(-1) # 3dim
        yyy = y[None, :, None].repeat(x.shape[0], 1, z.shape[0]).view(-1)
        zzz = z[None, None, :].repeat(x.shape[0], y.shape[0], 1).view(-1)

        if row_major:
            return xxx, yyy, zzz
        else:
            return yyy, xxx, zzz

    def grid_priors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): The device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, device=device)
            if self.verbose: print_tensor(f'[AnchorGen] level {i}', anchors)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self, featmap_size, level_idx, device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx].to(device)
        feat_h, feat_w, feat_d = featmap_size
        stride_w, stride_h, stride_d = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, device=device) * stride_w
        shift_y = torch.arange(0, feat_h, device=device) * stride_h
        shift_z = torch.arange(0, feat_d, device=device) * stride_d

        shift_xx, shift_yy, shift_zz = self._meshgrid(shift_x, shift_y, shift_z)
        shifts = torch.stack([shift_xx, shift_yy, shift_zz, shift_xx, shift_yy, shift_zz], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 6) to K shifts (K, 1, 6) to get
        # shifted anchors (K, A, 6), reshape to (K*A, 6)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 6)
        # first A rows correspond to A anchors of (0, 0, 0) in feature map,
        # then (0, 0, 1), (0, 0, 2), ...
        
        # print_tensor(f'[AnchorGen] l {level_idx} stride {self.strides[level_idx]} feat {featmap_size} ', all_anchors)
        # print_tensor(f'[AnchorGen] base anchor', base_anchors)
        return all_anchors

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse anchors according to the ``prior_idxs``.
        TODO
        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """

        height, width, depth = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs //
             num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width //
             num_base_anchors) % height * self.strides[level_idx][1]
        priors = torch.stack([x, y, x, y], 1).to(dtype).to(device) + \
            self.base_anchors[level_idx][base_anchor_id, :].to(device)

        return priors


    def valid_flags(self, featmap_sizes, pad_shape, device='cuda', verbose = False):
        """Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w, feat_d = featmap_sizes[i]
            h, w, d = list(pad_shape)[:3]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            valid_feat_d = min(int(np.ceil(d / anchor_stride[2])), feat_d)
            if verbose: print(f'[AnchorGen] level {i} stride {anchor_stride} feat shape {featmap_sizes[i]} img shape {pad_shape} ')
            flags = self.single_level_valid_flags((feat_h, feat_w, feat_d),
                                                  (valid_feat_h, valid_feat_w, valid_feat_d),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w, feat_d = featmap_size
        valid_h, valid_w, valid_d = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w and valid_d <= feat_d
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_z = torch.zeros(feat_d, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_z[:valid_d] = 1
        valid_xx, valid_yy, valid_zz = self._meshgrid(valid_x, valid_y, valid_z)
        valid = valid_xx & valid_yy & valid_zz
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str



@PRIOR_GENERATORS.register_module()
class YOLOAnchorGenerator3D(AnchorGenerator3D):
    """Anchor generator for YOLO.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """

    def __init__(self, strides, base_sizes):
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level, center=None):
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int, int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors

    def responsible_flags(self, featmap_sizes, gt_bboxes, device='cuda'):
        """Generate responsible anchor flags of grid cells in multiple scales.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in multiple
                feature levels.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): responsible flags of anchors in multiple level
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_responsible_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            flags = self.single_level_responsible_flags(
                featmap_sizes[i],
                gt_bboxes,
                anchor_stride,
                self.num_base_anchors[i],
                device=device)
            multi_level_responsible_flags.append(flags)
        return multi_level_responsible_flags

    def single_level_responsible_flags(self,
                                       featmap_size,
                                       gt_bboxes,
                                       stride,
                                       num_base_anchors,
                                       device='cuda'):
        """Generate the responsible flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            gt_bboxes (Tensor): Ground truth boxes, shape (n, 4).
            stride (tuple(int)): stride of current level
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = ((gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5).to(device)
        gt_bboxes_cy = ((gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5).to(device)
        gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / stride[0]).long()
        gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / stride[1]).long()

        # row major indexing
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x

        responsible_grid = torch.zeros(
            feat_h * feat_w, dtype=torch.uint8, device=device)
        responsible_grid[gt_bboxes_grid_idx] = 1

        responsible_grid = responsible_grid[:, None].expand(
            responsible_grid.size(0), num_base_anchors).contiguous().view(-1)
        return responsible_grid
