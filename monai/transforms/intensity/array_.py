"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

"""
A collection of "vanilla" transforms for intensity adjustment
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import numpy as np
from typing import Any, Optional, Sequence, Tuple, Union
from monai.transforms.compose import Randomizable, Transform
from monai.utils import dtype_numpy_to_torch
import torch

class RandGaussianNoise_(Randomizable, Transform):
    """
    Add Gaussian noise to image. No need to transform to CPU, original official function will transform to CPU and transform back to GPU after adding noise

    Args:
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
    """

    def __init__(self, prob: float = 0.1, mean: Union[Sequence[float], float] = 0.0, std: float = 0.1) -> None:
        self.prob = prob
        self.mean = mean
        self.std = std
        self._do_transform = False
        self._noise = None

    def randomize(self, im_shape: Sequence[int]) -> None:
        self._do_transform = self.R.random() < self.prob
        self._noise = torch.normal(mean=self.mean, std=self.R.uniform(0, self.std), size=im_shape)

    def __call__(self, img: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply the transform to `img`.
        """
        self.randomize(img.shape)
        assert self._noise is not None
        if not self._do_transform:
            return img
        dtype = dtype_numpy_to_torch(img.dtype) if isinstance(img, torch.Tensor) else img.dtype
        return img + self._noise.type(dtype)


class NNUNetNormalizeIntensity(Transform):
    """
    Normalize input based on provided args with nnUNet method, crop intensity range within [0.5, 99.5],
    using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        subtrahend: Optional[Union[Sequence, float]] = None,
        divisor: Optional[Union[Sequence, float]] = None,
        percentile_99_5: Optional[Union[Sequence, float]] = None,
        percentile_00_5: Optional[Union[Sequence, float]] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.percentile_99_5 = percentile_99_5
        self.percentile_00_5 = percentile_00_5
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img: Union[np.ndarray, torch.Tensor], sub=None, div=None, percentile_99_5=None, percentile_00_5=None) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(img, np.ndarray):
            slices = (img != 0) if self.nonzero else np.ones(img.shape, dtype=np.bool_)
            if not np.any(slices):
                return img

            _sub = sub if sub is not None else np.mean(img[slices])
            if isinstance(_sub, np.ndarray):
                _sub = _sub[slices]

            _div = div if div is not None else np.std(img[slices])
            if np.isscalar(_div):
                if _div == 0.0:
                    _div = 1.0
            elif isinstance(_div, np.ndarray):
                _div = _div[slices]
                _div[_div == 0.0] = 1.0
            if percentile_99_5 is not None and percentile_00_5 is not None:
                img[slices] = np.clip(img[slices], percentile_00_5, percentile_99_5)
            img[slices] = (img[slices] - _sub) / _div
            return img
        elif isinstance(img, torch.Tensor):
            slices = (img != 0) if self.nonzero else torch.ones(img.shape, dtype=torch.bool)
            if not torch.any(slices):
                return img

            _sub = sub if sub is not None else torch.mean(img[slices])
            if isinstance(_sub, torch.Tensor):
                _sub = _sub[slices]

            _div = div if div is not None else torch.std(img[slices])
            if np.isscalar(_div):
                if _div == 0.0:
                    _div = 1.0
            elif isinstance(_div, torch.Tensor):
                _div = _div[slices]
                _div[_div == 0.0] = 1.0
            if percentile_99_5 is not None and percentile_00_5 is not None:
                img[slices] = torch.clip(img[slices], percentile_00_5, percentile_99_5)
            img[slices] = (img[slices] - _sub) / _div
            return img
        else:
            raise ValueError('input is not tensor or numpy')

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components.")
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.divisor)} components.")
            if self.percentile_99_5 is not None and len(self.percentile_99_5) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.percentile_99_5)} components.")
            if self.percentile_00_5 is not None and len(self.percentile_00_5) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.percentile_00_5)} components.")

            for i, d in enumerate(img):
                img[i] = self._normalize(
                    d,
                    sub=self.subtrahend[i] if self.subtrahend is not None else None,
                    div=self.divisor[i] if self.divisor is not None else None,
                    percentile_99_5=self.percentile_99_5 if self.percentile_99_5 is not None else None,
                    percentile_00_5=self.percentile_00_5 if self.percentile_00_5 is not None else None,
                )
        else:
            img = self._normalize(img, self.subtrahend, self.divisor, self.percentile_99_5, self.percentile_00_5)

        return img


class AdjustWWWL(Transform):
    """
    Adjust intensity uniformly for the entire image with specified `window width and window level`.

    Args:
        ww: window width value to shift the intensity of image.
        wl: window level value to shift the intensity of image.
    """

    def __init__(self, ww: float, wl: float) -> None:
        self.ww = ww
        self.wl = wl

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`.
        """
        minv = self.wl - self.ww/2
        maxv = self.wl + self.ww/2
        img[img < minv] = minv
        img[img > maxv] = maxv
        return img


class RandAdjustWWWL(Randomizable, Transform):
    """
    Randomly adjust intensity with randomly picked window width and window level.

    Args:
        ww_range: random window width range.
        wl_range: random window level range.
        prob: probability of adjust wwwl.
    """

    def __init__(
            self,
            ww_range: Tuple[float, float],
            wl_range: Tuple[float, float],
            prob: float = 0.1) -> None:

        self.ww_range = ww_range
        self.wl_range = wl_range
        self.prob = prob
        self._do_transform = False

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob
        self.ww = self.R.uniform(low=self.ww_range[0], high=self.ww_range[1])
        self.wl = self.R.uniform(low=self.wl_range[0], high=self.wl_range[1])

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        self.randomize()
        if not self._do_transform:
            return img
        return AdjustWWWL(self.ww, self.wl)(img)



class NormalizeIntensityGPU(Transform):
    """
    Normalize input based on provided args with nnUNet method, crop intensity range within [0.5, 99.5],
    using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        percentile_99_5: the maximum intensity values.
        percentile_00_5: the minimum intensity values.
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
        self,
        subtrahend: Optional[Union[Sequence, float]] = None,
        divisor: Optional[Union[Sequence, float]] = None,
        percentile_99_5: Optional[Union[Sequence, float]] = None,
        percentile_00_5: Optional[Union[Sequence, float]] = None,
        nonzero: bool = False,
        channel_wise: bool = False,
    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.percentile_99_5 = percentile_99_5
        self.percentile_00_5 = percentile_00_5
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def _normalize(self, img: Union[np.ndarray, torch.Tensor], sub=None, div=None, percentile_99_5=None, percentile_00_5=None) -> Union[np.ndarray, torch.Tensor]:
        if div == 0.0: div = 1.0
        if percentile_99_5 is not None and percentile_00_5 is not None:
            img = torch.clip(img, percentile_00_5, percentile_99_5)
        img = (img - sub) / div
        return img

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        img = self._normalize(img, self.subtrahend, self.divisor, self.percentile_99_5, self.percentile_00_5)
        return img