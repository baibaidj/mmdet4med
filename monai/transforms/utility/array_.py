"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import TypeVar

import numpy as np
import torch

from monai.transforms.compose import Transform

# Generic type which can represent either a numpy.ndarray or a torch.Tensor
# Unlike Union can create a dependence between parameter(s) / return(s)
NdarrayTensor = TypeVar("NdarrayTensor", np.ndarray, torch.Tensor)



class SelectChannel(Transform):
    """
    select some channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """


    def __init__(self, channel_dim: int = 0) -> None:
        assert isinstance(channel_dim, int) and channel_dim >= 0, "invalid channel dimension."
        self.channel_dim = channel_dim

    def __call__(self, img: NdarrayTensor) -> NdarrayTensor:
        """
        Apply the transform to `img`.
        """
        assert self.channel_dim < img.shape[0]
        img = img[self.channel_dim]
        return img[None]