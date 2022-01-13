"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Union, Tuple

import numpy as np
import torch

from monai.config import KeysCollection, DtypeLike
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.spatial.array_ import (Flip_, DimensionReduction, Rotate90Flip_, ResampleGPU, RandAffineGridGPU, totensor)
from monai.transforms.utils_ import create_grid_torch
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)

from monai.transforms.spatial.array import Spacing, Rand3DElastic
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from mmcv import Timer
import pdb
from ..utils_ import print_tensor

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]



class RandRotate90Flipd_(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate90Flip`.
    With probability `prob`, input arrays are rotated by 90 degrees and flipped in 3 axes
    in the plane specified by `spatial_axes`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
        """
        super().__init__(keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k

        self._do_transform = False
        self._rand_k = 0

        self.euler_flips = np.array([[0, 0, 0, -1, -1, -1],
                                [0, 0, 0, -1, -1, 1],
                                [90, 0, 0, -1, 1, -1],
                                [90, 0, 0, -1, 1, 1],
                                [90, 0, 0, -1, -1, -1],
                                [90, 0, 0, -1, -1, 1],
                                [0, 0, 0, -1, 1, -1],
                                [0, 0, 0, -1, 1, 1],
                                [0, 0, 90, 1, -1, -1],
                                [0, 0, 90, 1, -1, 1],
                                [0, 90, 90, 1, -1, 1],
                                [0, 90, 90, 1, -1, -1],
                                [0, 90, 90, 1, 1, 1],
                                [0, 90, 90, 1, 1, -1],
                                [0, 0, 90, 1, 1, -1],
                                [0, 0, 90, 1, 1, 1],
                                [90, 0, 90, -1, -1, -1],
                                [90, 0, 90, -1, -1, 1],
                                [0, 90, 0, -1, -1, 1],
                                [0, 90, 0, -1, -1, -1],
                                [0, 90, 0, -1, 1, 1],
                                [0, 90, 0, -1, 1, -1],
                                [90, 0, 90, -1, 1, -1],
                                [90, 0, 90, -1, 1, 1],
                                [90, 0, 90, 1, -1, -1],
                                [90, 0, 90, 1, -1, 1],
                                [0, 90, 0, 1, -1, 1],
                                [0, 90, 0, 1, -1, -1],
                                [0, 90, 0, 1, 1, 1],
                                [0, 90, 0, 1, 1, -1],
                                [90, 0, 90, 1, 1, -1],
                                [90, 0, 90, 1, 1, 1],
                                [0, 0, 90, -1, -1, -1],
                                [0, 0, 90, -1, -1, 1],
                                [0, 90, 90, -1, -1, 1],
                                [0, 90, 90, -1, -1, -1],
                                [0, 90, 90, -1, 1, 1],
                                [0, 90, 90, -1, 1, -1],
                                [0, 0, 90, -1, 1, -1],
                                [0, 0, 90, -1, 1, 1],
                                [0, 0, 0, 1, -1, -1],
                                [0, 0, 0, 1, -1, 1],
                                [90, 0, 0, 1, 1, -1],
                                [90, 0, 0, 1, 1, 1],
                                [90, 0, 0, 1, -1, -1],
                                [90, 0, 0, 1, -1, 1],
                                [0, 0, 0, 1, 1, -1],
                                [0, 0, 0, 1, 1, 1]], dtype="float32")

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob
        idx = np.random.permutation(self.euler_flips.shape[0])[0]
        self._affine = self.euler_flips[idx]

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Mapping[Hashable, Union[np.ndarray, torch.Tensor]]:
        self.randomize()
        if not self._do_transform:
            return data

        rotator_flipper = Rotate90Flip_(self._affine)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator_flipper(d[key])
        return d


class RandFlipd_(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip_`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__(keys)
        self.spatial_axis = spatial_axis
        self.prob = prob

        self._do_transform = False
        self.flipper = Flip_(spatial_axis=spatial_axis)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


class FlipTTAd_(MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip_`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 1.0,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__(keys)

        self.spatial_axis = spatial_axis

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        # self.randomize()
        d = dict(data)
        flip = d['img_meta_dict']['flip']
        flip_direction = d['img_meta_dict']['flip_direction']

        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None
        # print(f'PIPELINE: Flip {flip} Direction {flip_direction} dims {flip_dims}')
        if flip and flip_dims is not None: 
            flipper = Flip_(flip_dims)
            for key in self.keys:
                d[key] = flipper(d[key])
        return d



class DimensionReductiond(MapTransform):
    """
    cast a 3D image to batches of 2D image.
    Dictionary-based version :py:class:`monai.transforms.DimensionReductiond`.

    Args:
        keys: Keys to pick data for transformation.
        reduction_dim: Which dimention need to reduction.
        context: How many neighbours to need.
    """
    def __init__(self, keys: KeysCollection, reduction_dim: int,
                 context: Mapping[str, int]) -> None:
        super().__init__(keys)
        self.reductiom_dim = reduction_dim
        self.context = context
        self.reductiom_fun = DimensionReduction()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        for key in self.keys:
            data[key] = self.reductiom_fun(data[key], self.reductiom_dim,
                                           self.context[key])
        return data


class RandSelectDimd(MapTransform, Randomizable):
    """
    Random select number of slices from specified dimention.
    Dictionary-based version :py:class:`monai.transforms.RandSelectDimd`.

    Args:
        keys: Keys to pick data for transformation.
        dim: Which dimention need to select from.
        size: How many slice need to select.
    """
    def __init__(
        self,
        keys: KeysCollection,
        dim: int,
        size: int,
    ) -> None:
        super().__init__(keys)
        self.size = size
        self.dim = dim
        self.selected = None

    def randomize(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        n = data.shape[self.dim]
        self.selected = np.random.choice(n, self.size)

    def select_dimention(self, data):
        shape = data.shape
        slices = [slice(0, s) for s in shape]
        result = []
        for i in self.selected:
            slices[self.dim] = slice(i, i + 1)
            result.append(data[tuple(slices)])
        return np.concatenate(result, self.dim)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        self.randomize(data[self.keys[0]])
        for key in self.keys:
            data[key] = self.select_dimention(data[key])
        return data


class RandCropDimd(MapTransform, Randomizable):
    """
    Random crop on a specified dimention.
    Dictionary-based version :py:class:`monai.transforms.RandCropDimd`.

    Args:
        keys: Keys to pick data for transformation.
        dim: The dimention to crop
        size: The target size.
    """
    def __init__(
        self,
        keys: KeysCollection,
        dim: int,
        size: int,
    ) -> None:
        super().__init__(keys)
        self.size = size
        self.dim = dim
        self.center = None

    def randomize(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        rng = range(self.size // 2, data.shape[self.dim] - self.size // 2)
        self.center = np.random.choice(rng)

    def crop_dimention(self, data):
        shape = data.shape
        slices = [slice(0, s) for s in shape]
        start = self.center - self.size // 2
        slices[self.dim] = slice(start, start + self.size)
        return data[tuple(slices)]

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        self.randomize(data[self.keys[0]])
        for key in self.keys:
            data[key] = self.crop_dimention(data[key])
        return data



class SpacingTTAd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Sequence[float],
        diagonal: bool = False,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Optional[Union[Sequence[DtypeLike], DtypeLike]] = np.float64,
        meta_key_postfix: str = "meta_dict",
        verbose = False,

    ) -> None:
        """
        Args:
            pixdim: output voxel spacing.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__(keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.diagonal = diagonal
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.verbose = verbose

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        new_pixdim = d['img_meta_dict'].get('new_pixdim', None)
        # print('Check new pixdim', new_pixdim)
        if new_pixdim is None: return d
        self.spacing_transform = Spacing(new_pixdim, diagonal=self.diagonal, dtype = np.float32)

        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            old_shape = d[key].shape; old_spacing = [meta_data['affine'][i, i] for i in range(3)]
            d[key], _, new_affine = self.spacing_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
            )
            new_shape = d[key].shape[-3:]
            if self.verbose: print(f'\tRespacing: from {old_shape} {old_spacing} to {new_shape} {new_pixdim}')
            # set the 'affine' key
            meta_data["affine"] = new_affine
            meta_data['shape_post_resize'] = new_shape
        return d

class Rand3DElasticGPUd(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        verbose = False, 
        meta_key = 'img_meta_dict',
    ) -> None: 
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
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
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        super().__init__(keys)
        # rotate_angle * np.pi / 180.0
        rotate_range = [a * np.pi / 180.0 for a in rotate_range]
        
        self.rand_affine_grid = RandAffineGridGPU(rotate_range, shear_range, translate_range, scale_range, device)
        self.resampler = ResampleGPU(as_tensor_output=as_tensor_output, device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        # self.mode: GridSampleMode = GridSampleMode(mode)
        # self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.device = device

        self.prob = prob
        self.do_transform = False
        self.rand_offset = None
        self.magnitude = 1.0
        self.sigma = 1.0
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        # self.grid_tensor = create_grid_torch(spatial_size=spatial_size).half()
        self.grid_tensor = totensor(create_grid(spatial_size)).float() #.half()
        self.verbose = verbose
        self.meta_key = meta_key

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.do_transform = self.R.rand() < self.prob
        # if self.do_transform:
            # self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32)
        self.unidist = torch.distributions.uniform.Uniform(-1.0, 1.0)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        sp_size = fall_back_tuple(self.spatial_size, data[self.keys[0]].shape[-3:])
        self.randomize(grid_size=sp_size)

        device = d[self.keys[0]].device
        dtype = torch.float if device.type == 'cpu' else torch.half
        if self.verbose: print(f'Device:{device} UseDtype:{dtype}')
        key2dtype = {k : d[k].dtype for k in self.keys}
        # print('ELASTIC', device)
        if self.do_transform:
            # with Timer(print_tmpl='\tElasticTrans {:.3f} seconds'):
            self.grid_tensor = self.grid_tensor.to(device)
            grid4d = self.grid_tensor.clone().to(dtype)
            sigma = torch.tensor(self.sigma, device = device).to(dtype)
            gaussian = GaussianFilter(spatial_dims=3, sigma=sigma, truncated=3.0).to(device) # larger sigma, more flattening curve
            offset4d = self.unidist.rsample([3] + list(sp_size)).to(dtype).to(device).unsqueeze(0)  # (-1.0, 1.0)
            offset_gaus = gaussian(offset4d)[0] # -0.02, 0.02
            grid4d[:3] = grid4d[:3] + offset_gaus * self.magnitude # jitter 1 to 2 depend on the magnitude

            if self.verbose: print_tensor('[RElast] initial grid', self.grid_tensor)
            if self.verbose: print_tensor('[RElast]offset uniform', offset4d)
            if self.verbose: print_tensor(f'[RElast]offset gaussian sigma:{self.sigma} mag:{self.magnitude}', offset_gaus)
            if self.verbose: print_tensor('[RElast]offset grid', grid4d[:3])
            # pdb.set_trace()
            grid4d = self.rand_affine_grid(grid=grid4d)
            if self.verbose: print_tensor('affine grid', grid4d)
            for idx, key in enumerate(self.keys):
                # with Timer(print_tmpl='\tElasticTransform %s {:.3f} seconds' %key):
                if self.verbose: print_tensor(f'[RElast] {key} prev', d[key])
                d[key] = self.resampler(d[key], grid4d, mode=self.mode[idx], 
                                        padding_mode=self.padding_mode[idx])
                d[key] = d[key].to(key2dtype[key])
                if self.verbose: print_tensor(f'[RElast] {key} after', d[key])
                # print_tensor(f'[RandElastic] {key}', d[key])
                for bi, meta_d in enumerate(d[self.meta_key]): meta_d['patch_shape'] = tuple(d[key].shape[-3:])
        return d

