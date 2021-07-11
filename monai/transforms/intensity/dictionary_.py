"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

"""
A collection of dictionary-based wrappers around the "vanilla" transforms for intensity adjustment
defined in :py:class:`monai.transforms.intensity.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from typing import Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.intensity.array_ import (
    NNUNetNormalizeIntensity,
    AdjustWWWL,
    RandAdjustWWWL,
    NormalizeIntensityGPU
)
# from monai.transfroms import print_tensor
from monai.utils import dtype_numpy_to_torch, ensure_tuple_size
from mmcv import Timer
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

class RandGaussianNoised_(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise_`.
    Add Gaussian noise to image. This transform assumes all the expected fields have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(
            self, keys: KeysCollection, prob: float = 0.1, mean: Union[Sequence[float], float] = 0.0, std: float = 0.1
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.mean = ensure_tuple_size(mean, len(self.keys))
        self.std = std
        self._do_transform = False
        self._noise: Optional[np.ndarray] = None

    def randomize(self) -> None:
        self._do_transform = self.R.random() < self.prob
    
    def randnoise(self, im_shape: Sequence[int], device: torch.device):
        return torch.normal(mean=self.mean[0], std=self.R.uniform(0, self.std), size=im_shape).to(device)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        # with Timer(print_tmpl='\tRandNoise %s {:.3f} seconds'): #TODO
        if not self._do_transform:
            return d
        else:
            image_shape = d[self.keys[0]].shape  # image shape from the first data key
            rand_noise_tensor = self.randnoise(image_shape, d[self.keys[0]].device)
            for key in self.keys:
                dtype = dtype_numpy_to_torch(d[key].dtype) if isinstance(d[key], np.ndarray) else d[key].dtype
                d[key] = d[key] + rand_noise_tensor.to(dtype)
        return d


class NNUNetNormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NNUNetNormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
            self,
            keys: KeysCollection,
            subtrahend: Optional[Union[Sequence, float]] = None,
            divisor: Optional[Union[Sequence, float]] = None,
            percentile_99_5: Optional[Union[Sequence, float]] = None,
            percentile_00_5: Optional[Union[Sequence, float]] = None,
            nonzero: bool = False,
            channel_wise: bool = False,
    ) -> None:
        super().__init__(keys)
        self.normalizer = NNUNetNormalizeIntensity(subtrahend, divisor, percentile_99_5, percentile_00_5, nonzero, channel_wise)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        with Timer(print_tmpl='\tNORMIntensity {:.3f} seconds'): #TODO
            for key in self.keys:
                d[key] = self.normalizer(d[key])
        return d


class AdjustWWWLd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AdjustWWWL`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        ww: window width value to shift the intensity of image.
        wl: window level value to shift the intensity of image.
    """

    def __init__(self, keys: KeysCollection, ww: float, wl: float) -> None:
        super().__init__(keys)
        self.wwwler = AdjustWWWL(ww, wl)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.wwwler(d[key])
        return d


class RandAdjustWWWLd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAdjustWWWL`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        ww_range: the range of window width value to shift the intensity of image.
        wl_range: the range of window level value to shift the intensity of image.
    """

    def __init__(
            self,
            keys: KeysCollection,
            ww_range: Tuple[float, float],
            wl_range: Tuple[float, float],
            prob: float = 0.1
    ) -> None:
        super().__init__(keys)
        self.converter = RandAdjustWWWL(ww_range, wl_range, prob)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


NNUNetNormalizeIntensityD = NNUNetNormalizeIntensityDict = NNUNetNormalizeIntensityd
AdjustWWWLD = AdjustWWWLDict = AdjustWWWLd
RandAdjustWWWLD = RandAdjustWWWLDict = RandAdjustWWWLd


class ConvertLabeld(MapTransform):

    def __init__(
        self,
        keys = ['label'],
        # limit_label_to = None, 
        label_mapping = None,
        ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            limit_label_to: [Union(None, list/tuple)]only reserve labels of specified values
            label_mapping: Dict, convert label values for training

        """
        super().__init__(keys)
        self.label_mapping = label_mapping

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            with Timer(print_tmpl='\tConvertLabel {:.3f} seconds'): #TODO
                d[key] = convert_label(d[key], label_mapping=self.label_mapping)
        return d


def convert_label(label, label_mapping = None, inverse=False):
    if label_mapping is None:
        return label
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    
    if not inverse:
        max_value = max([v for _, v in label_mapping.items()])
        label[label > max_value] = 0
    return label


class ConvertLabelRadiald(MapTransform):

    def __init__(
        self,
        keys = ['label'],
        # limit_label_to = None, 
        label_mapping = None,
        verbose = False
        ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            limit_label_to: [Union(None, list/tuple)]only reserve labels of specified values
            label_mapping: Dict, convert label values for training

        """
        super().__init__(keys)
        self.label_mapping = label_mapping
        self.verbose = verbose
        # self.limit_label_to = limit_label_to

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        # with Timer(print_tmpl='\tConvertLabelRadial {:.3f} seconds'): #TODO
        for key in self.keys: 
            d[key] = radialdist_regroup(d[key], self.label_mapping, verbose= self.verbose)
        return d
    
def radialdist_regroup(radial_dist_map, label_mapping, verbose = True):
    if verbose: print_tensor('\nRegroup: start', radial_dist_map)
    # new_pred = radial_dist_map.new_empty(radial_dist_map.shape)
    for c, g in label_mapping.items():
        tic, toc = c, c + 1
        if c == 0: toc = c + 0.9999
        mask_c = ((radial_dist_map > tic) & (radial_dist_map <= toc)).bool() #.astype(radial_dist_map.dtype)
        radial_dist_map[mask_c] = (radial_dist_map[mask_c] - c ) + g
        # if verbose: print_tensor(f'\tClassRange {tic}-{toc} to {g}', radial_dist_map[mask_c])
    if verbose: print_tensor('Regroup: final', radial_dist_map)
    return radial_dist_map


class NormalizeIntensityGPUd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NNUNetNormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
            self,
            keys: KeysCollection,
            subtrahend: Optional[Union[Sequence, float]] = None,
            divisor: Optional[Union[Sequence, float]] = None,
            percentile_99_5: Optional[Union[Sequence, float]] = None,
            percentile_00_5: Optional[Union[Sequence, float]] = None,
            nonzero: bool = False,
            channel_wise: bool = False,
    ) -> None:
        super().__init__(keys)
        self.normalizer = NormalizeIntensityGPU(subtrahend, divisor, percentile_99_5, percentile_00_5, nonzero, channel_wise)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        # with Timer(print_tmpl='\tNORMIntensity {:.3f} seconds'): #TODO
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d
