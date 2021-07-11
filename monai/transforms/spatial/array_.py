"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch, pdb
from monai.transforms.compose import Transform, Randomizable
from monai.transforms.spatial.array import Flip, Resample
import warnings

from monai.transforms.utils import (
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
)
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]
from ..utils_ import print_tensor


class Rotate90Flip_(Transform):
    """
    Rotate and Reverses the order of elements along the all spatial axis. Preserves shape.
    Uses ``np.flip`, 'np.rotate'` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    """
    def __init__(self, affine) -> None:
        self.affine = affine

    def __call__(
        self, img: Union[np.ndarray,
                         torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """


        roll, pitch, yaw, fx, fy, fz = self.affine
        if isinstance(img, torch.Tensor):
            img = torch.rot90(img, roll // 90, dims=(2, 3))
            img = torch.rot90(img, pitch // 90, dims=(3, 1))
            img = torch.rot90(img, yaw // 90, dims=(1, 2))
        elif isinstance(img, np.ndarray):
            img = np.rot90(img, roll // 90, axes=(2, 3))
            img = np.rot90(img, pitch // 90, axes=(3, 1))
            img = np.rot90(img, yaw // 90, axes=(1, 2))
        if fx == -1:
            img = img[:, ::-1, :, :]
        if fy == -1:
            img = img[:, :, ::-1, :]
        if fz == -1:
            img = img[:, :, :, ::-1]
        if isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img)
        return img


class Flip_(Transform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
    """
    def __init__(
            self, spatial_axis: Optional[Union[Sequence[tuple],
                                               tuple]]) -> None:
        self.spatial_axis = spatial_axis

    def __call__(
        self, img: Union[np.ndarray,
                         torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        flipped = list()
        if isinstance(img, torch.Tensor):
            for channel in img:
                flipped.append(torch.flip(channel, self.spatial_axis))
            return torch.stack(flipped)
        elif isinstance(img, np.ndarray):
            f = Flip(self.spatial_axis)
            return f(img)


class DimensionReduction(Transform):
    """
    Reduction Image Dimention, for example: [chanel, H, W, D] -> [batch/(D), channel, H, W]

    Args:
    """
    def __init__(self) -> None:
        pass

    def __call__(self,
                 data: np.ndarray,
                 reduction_dim: int,
                 context: int, *args, **kwargs) -> np.ndarray:
        """
        cast a 3D image to batches of 2D image.
        Args:
            data: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            reduction_dim: which dimention need to reduction.
            context: how many neighbours to need.

        """
        shape = data.shape
        batch = shape[reduction_dim]
        slices = [slice(0, s) for s in shape]
        result = []
        for b in range(batch):
            start = b - context // 2
            stop = start + context
            start = max(0, start)
            stop = min(batch, stop)
            slices[reduction_dim] = slice(start, stop)
            patch = []
            for c in range(shape[0]):
                slices[0] = slice(c, c + 1)
                piece = data[tuple(slices)]
                rest = context - piece.shape[reduction_dim]
                if rest > 0:
                    if start == 0:
                        slices[reduction_dim] = slice(0, 1)
                        piece = [piece[tuple(slices)]] * rest + [piece]
                    else:
                        slices[reduction_dim] = slice(-1, None)
                        piece = [piece] + [piece[tuple(slices)]] * rest
                    piece = np.concatenate(piece, axis=reduction_dim)
                patch.append(piece)
            patch = np.concatenate(patch, axis=reduction_dim)
            patch = np.swapaxes(patch, 0, reduction_dim)
            patch = np.squeeze(patch, reduction_dim)
            result.append(patch)
        result = np.array(result)
        return result


class AffineGrid_(Transform):
    """
        Affine transforms on the coordinates.

        Args:
            affine_matrix: should be a homogeneous matrix, 2d is 3*3 and 3d is 4*4, e.g.: np.eye(4)
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.

        """

    def __init__(
            self,
            affine_matrix: Union[Sequence[float]] = None,
            as_tensor_output: bool = True,
            device: Optional[torch.device] = None,
    ) -> None:
        self.affine_matrix = affine_matrix
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
            self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")

        affine = torch.as_tensor(np.ascontiguousarray(self.affine_matrix), device=self.device)

        grid = torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
        grid = (affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if grid is None or not isinstance(grid, torch.Tensor):
            raise ValueError("Unknown grid.")
        if self.as_tensor_output:
            return grid
        return np.asarray(grid.cpu().numpy())


class Affine_(Transform):
    """
    Transform ``img`` given the affine parameters.
    """

    def __init__(
        self,
        affine_matrix: Union[Sequence[float]] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            affine_matrix: should be a homogeneous matrix, 2d is 3*3 and 3d is 4*4, e.g.: np.eye(4),
                note that, this affine_matrix is from origin to target transform, rotation center is (0, 0, 0)
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
        """
        # self.affine_grid = AffineGrid_(
        #     affine_matrix=affine_matrix,
        #     as_tensor_output=True,
        #     device=device,
        # )
        self.affine_matrix = affine_matrix
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.device = device

    def auto_set_outshape(
            self,
            img: Union[np.ndarray, torch.Tensor],
            affine_matrix: Union[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            affine_matrix: should be a homogeneous matrix, 2d is 3*3 and 3d is 4*4, e.g.: np.eye(4)
        """
        dim = len(img.shape) - 1
        affine_matrix = np.asmatrix(affine_matrix)
        coords = []
        if dim == 2:
            coord_x0_y0 = [0, 0, 0]
            coord_x1_y0 = [img.shape[1]-1, 0, 0]
            coord_x0_y1 = [0, img.shape[2]-1, 0]
            coord_x1_y1 = [img.shape[1], img.shape[2]-1, 0]
            coords.append(coord_x0_y0)
            coords.append(coord_x1_y0)
            coords.append(coord_x0_y1)
            coords.append(coord_x1_y1)

            minx, maxx, miny, maxy = 0, 0, 0, 0
            initialize = True
            for coord in coords:
                new_coord = np.dot(affine_matrix, np.asmatrix(coord).T)
                if initialize:
                    minx = new_coord[0, 0]
                    miny = new_coord[1, 0]
                    maxx = new_coord[0, 0]
                    maxy = new_coord[1, 0]
                    initialize = False
                else:
                    minx = min(minx, new_coord[0, 0])
                    miny = min(miny, new_coord[1, 0])
                    maxx = max(maxx, new_coord[0, 0])
                    maxy = max(maxy, new_coord[1, 0])
            spatial_size = np.array([maxx - minx + 1, maxy - miny + 1])
            spatial_size = np.round(spatial_size).astype(np.int)
            return spatial_size

        elif dim == 3:
            coord_x0_y0_z0 = [0, 0, 0, 0]
            coord_x1_y0_z0 = [img.shape[1]-1, 0, 0, 0]
            coord_x0_y1_z0 = [0, img.shape[2]-1, 0, 0]
            coord_x1_y1_z0 = [img.shape[1]-1, img.shape[2]-1, 0, 0]
            coord_x0_y0_z1 = [0, 0, img.shape[3]-1, 0]
            coord_x1_y0_z1 = [img.shape[1]-1, 0, img.shape[3]-1, 0]
            coord_x0_y1_z1 = [0, img.shape[2]-1, img.shape[3]-1, 0]
            coord_x1_y1_z1 = [img.shape[1]-1, img.shape[2]-1, img.shape[3]-1, 0]
            coords.append(coord_x0_y0_z0)
            coords.append(coord_x1_y0_z0)
            coords.append(coord_x0_y1_z0)
            coords.append(coord_x1_y1_z0)
            coords.append(coord_x0_y0_z1)
            coords.append(coord_x1_y0_z1)
            coords.append(coord_x0_y1_z1)
            coords.append(coord_x1_y1_z1)

            minx, maxx, miny, maxy, minz, maxz = 0, 0, 0, 0, 0, 0
            initialize = True
            for coord in coords:
                new_coord = np.dot(affine_matrix, np.asmatrix(coord).T)
                if initialize:
                    minx = new_coord[0, 0]
                    miny = new_coord[1, 0]
                    minz = new_coord[2, 0]
                    maxx = new_coord[0, 0]
                    maxy = new_coord[1, 0]
                    maxz = new_coord[2, 0]
                    initialize = False
                else:
                    minx = min(minx, new_coord[0, 0])
                    miny = min(miny, new_coord[1, 0])
                    minz = min(minz, new_coord[2, 0])
                    maxx = max(maxx, new_coord[0, 0])
                    maxy = max(maxy, new_coord[1, 0])
                    maxz = max(maxz, new_coord[2, 0])
            spatial_size = np.array([maxx - minx + 1, maxy - miny + 1, maxz - minz + 1])
            spatial_size = np.round(spatial_size).astype(np.int)
            return spatial_size
        else:
            raise ValueError('Input wrong image shape')


    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        if spatial_size is None:
            spatial_size = self.auto_set_outshape(img, affine_matrix=self.affine_matrix)

        # transform to image center
        center1 = np.asmatrix([(img.shape[1]-1)/2.0, (img.shape[2]-1)/2.0, (img.shape[3]-1)/2.0]).T
        center2 = np.asmatrix([(spatial_size[0]-1)/2.0, (spatial_size[1]-1)/2.0, (spatial_size[2]-1)/2.0]).T
        affine_matrix_c = self.affine_matrix.copy()
        t1 = affine_matrix_c[0:3, 0:3] * center1 - center2
        affine_matrix_c[0:3, 3:4] += t1
        affine_grid = AffineGrid_(
            affine_matrix=np.linalg.inv(affine_matrix_c),
            as_tensor_output=True,
            device=self.device,
        )
        sp_size = fall_back_tuple(spatial_size.tolist() or self.spatial_size, img.shape[1:])
        grid = affine_grid(spatial_size=sp_size)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )

def totensor(x, d = None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif x is None:
        return None
    else:
        x = torch.as_tensor(x)
    if d: return x.to(d)
    else: return x


class AffineGridGPU(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: angle range in radians. rotate_params[0] with be used to generate the 1st rotation
            parameter from `uniform[-rotate_params[0], rotate_params[0])`. Similarly, `rotate_params[1]` and
            `rotate_params[2]` are used in 3D affine for the range of 2nd and 3rd axes.
        shear_params: shear_params[0] with be used to generate the 1st shearing parameter from
            `uniform[-shear_params[0], shear_params[0])`. Similarly, `shear_params[1]` to
            `shear_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        translate_params : translate_params[0] with be used to generate the 1st shift parameter from
            `uniform[-translate_params[0], translate_params[0])`. Similarly, `translate_params[1]`
            to `translate_params[N]` controls the range of the uniform distribution used to generate
            the 2nd to N-th parameter.
        scale_params: scale_params[0] with be used to generate the 1st scaling factor from
            `uniform[-scale_params[0], scale_params[0]) + 1.0`. Similarly, `scale_params[1]` to
            `scale_params[N]` controls the range of the uniform distribution used to generate the 2nd to
            N-th parameter.
        as_tensor_output: whether to output tensor instead of numpy array.
            defaults to True.
        device: device to store the output grid data.

    """

    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        # as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params

        # self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")
        
        device = getattr(grid, 'device') or self.device
        spatial_dims = len(grid.shape) - 1

        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            affine = affine @ create_rotate(spatial_dims, self.rotate_params)
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            affine = affine @ create_translate(spatial_dims, self.translate_params)
        if self.scale_params:
            affine = affine @ create_scale(spatial_dims, self.scale_params)
        # affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)
        # grid = torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid.detach().clone()
        grid = totensor(grid, device)
        affine = totensor(affine, device).to(grid.dtype)
        # print_tensor('\tgrid', grid)
        # print_tensor('\taffine', affine)
        grid = (affine @ grid.reshape((grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))
        return grid


class RandAffineGridGPU(Randomizable, Transform):
    """
    Generate randomised affine grid.
    """

    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        device = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        # self.as_tensor_output = as_tensor_output
        self.device = device

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGridGPU(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            # as_tensor_output=self.as_tensor_output,
            device=grid.device,
        )
        return affine_grid(spatial_size, grid)



class ResampleGPU(Transform):
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        verbose = False
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device: device on which the tensor will be allocated.
        """
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.as_tensor_output = as_tensor_output
        self.device = device
        self.verbose = verbose

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        assert isinstance(img, torch.Tensor)
        assert isinstance(grid, torch.Tensor)
        grid = torch.tensor(grid) if not isinstance(grid, torch.Tensor) else grid.detach().clone()
        has_batch_dim = img.dim() == 5

        for i, dim in enumerate(img.shape[-3:]):
            grid[i] = 2.0 * grid[i] / (dim - 1.0)
        
        if self.verbose: print_tensor('grid norm 3c', grid[:3])
        if self.verbose: print_tensor('grid norm 4d', grid[-1:])
        grid = grid[:-1] / grid[-1:] 
        
        if self.verbose: print_tensor('grid norm div', grid)

        index_ordering: List[int] = list(range(img.ndimension() - (3 if has_batch_dim else 2) , -1, -1))
        if self.verbose: print(index_ordering)
        grid = grid[index_ordering]
        grid = grid.permute(list(range(grid.ndimension()))[1:] + [0])
        
        if self.verbose: print_tensor('grid norm permute', grid)
        if self.verbose: print(f'mode {self.mode.value} pad {self.padding_mode.value}')
        if has_batch_dim:
            num_batch = img.shape[0]
            grid_shape = grid.shape
            grid = grid.unsqueeze(0).expand(num_batch, *grid_shape)
        else:
            img, grid = img.unsqueeze(0), grid.unsqueeze(0)
        
        out = torch.nn.functional.grid_sample(
            img.float(),
            grid.float(),
            mode=self.mode.value if mode is None else GridSampleMode(mode).value,
            padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
            align_corners=True)
        
        if self.verbose: print_tensor('input', img)
        if self.verbose: print_tensor('output', out)

        if not has_batch_dim: out = out[0]
        return out