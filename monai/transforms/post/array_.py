# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for the model output tensors
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks import one_hot
from monai.transforms.compose import Transform
from monai.transforms.utils import get_largest_connected_component_mask
from monai.transforms.utils_ import get_topn_connected_component_mask
from monai.utils import ensure_tuple

__all__ = [
    "KeepTopNConnectedComponent",
    "FilterConnectedComponent",
]


class KeepTopNConnectedComponent(Transform):
    """
    Similar to KeepLargestConnectedComponent but Keep the top n largest connected component in the image.
    """
    def __init__(self,
                 applied_labels: Union[Sequence[int], int],
                 independent: bool = True,
                 connectivity: Optional[int] = None,
                 topn: Union[Sequence[int], int] = 1,
                 topn_call_back=get_topn_connected_component_mask) -> None:
        """
        Args:
            applied_labels: Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is used to determine what channels to apply.
            independent: consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.
            topn: the top n largest connected components to keep. only applicable for independent=True.
        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels)
        self.independent = independent
        self.connectivity = connectivity

        self.topn = ensure_tuple(topn)
        if isinstance(topn, (int, np.integer)):
            self.topn = [topn] * len(self.applied_labels)
        assert len(self.topn) == len(
            self.applied_labels
        ), f'applied_labels and topn should have the same length.'

        self.topn_call_back = topn_call_back

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: shape must be (batch_size, C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            A PyTorch Tensor with shape (batch_size, C, spatial_dim1[, spatial_dim2, ...]).
        """
        channel_dim = 1
        if img.shape[channel_dim] == 1:

            img = torch.squeeze(img, dim=channel_dim)

            if self.independent:
                for i, n in zip(self.applied_labels, self.topn):
                    foreground = (img == i).type(torch.uint8)
                    mask = self.topn_call_back(foreground,
                                               self.connectivity,
                                               topn=n)
                    img[foreground != mask] = 0
            else:
                foreground = torch.zeros_like(img)
                for i in self.applied_labels:
                    foreground += (img == i).type(torch.uint8)
                mask = get_largest_connected_component_mask(
                    foreground, self.connectivity)
                img[foreground != mask] = 0
            output = torch.unsqueeze(img, dim=channel_dim)
        else:
            # one-hot data is assumed to have binary value in each channel
            if self.independent:
                for i, n in zip(self.applied_labels, self.topn):
                    foreground = img[:, i, ...].type(torch.uint8)
                    mask = self.topn_call_back(foreground,
                                               self.connectivity,
                                               topn=n)
                    img[:, i, ...][foreground != mask] = 0
            else:
                applied_img = img[:, self.applied_labels,
                                  ...].type(torch.uint8)
                foreground = torch.any(applied_img, dim=channel_dim)
                mask = get_largest_connected_component_mask(
                    foreground, self.connectivity)
                background_mask = torch.unsqueeze(foreground != mask,
                                                  dim=channel_dim)
                background_mask = torch.repeat_interleave(
                    background_mask, len(self.applied_labels), dim=channel_dim)
                applied_img[background_mask] = 0
                img[:, self.applied_labels, ...] = applied_img.type(img.type())
            output = img

        return output


class FilterConnectedComponent(Transform):
    """
    General Filter for connected component in the label.
    """
    def __init__(
        self,
        applied_labels: Union[Sequence[int], int],
        independent: bool = True,
        filter_callbacks: Union[Sequence[Callable],
                                Callable] = get_topn_connected_component_mask
    ) -> None:
        """
        Args:
            applied_labels: Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is used to determine what channels to apply.
            independent: consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            filter_callbacks: the filters for applied_labels.
        Note:
            independent is false can only remove component while true can add extra 
                component, fill hole for example.
        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels)
        self.independent = independent
        self.filter_callbacks = filter_callbacks

        if not independent:
            assert callable(
                filter_callbacks
            ), f'independent is False require only one call_back'
        else:
            if not isinstance(filter_callbacks, (list, tuple)):
                self.filter_callbacks = [filter_callbacks] * len(
                    self.applied_labels)
            assert len(self.filter_callbacks) == len(
                self.applied_labels
            ), f'applied_labels and filter_callbacks should have the same length.'

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: shape must be (batch_size, C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            A PyTorch Tensor with shape (batch_size, C, spatial_dim1[, spatial_dim2, ...]).
        """
        channel_dim = 1
        if img.shape[channel_dim] == 1:

            img = torch.squeeze(img, dim=channel_dim)

            if self.independent:
                for i, call_back in zip(self.applied_labels,
                                        self.filter_callbacks):
                    foreground = (img == i).type(torch.uint8)
                    mask = call_back(foreground)
                    img[foreground != mask] = 0
                    img[mask.type(torch.bool)] = i
            else:
                foreground = torch.zeros_like(img)
                for i in self.applied_labels:
                    foreground += (img == i).type(torch.uint8)
                mask = self.filter_callbacks[0](foreground)
                img[foreground != mask] = 0
            output = torch.unsqueeze(img, dim=channel_dim)
        else:
            # one-hot data is assumed to have binary value in each channel
            if self.independent:
                for i, call_back in zip(self.applied_labels,
                                        self.filter_callbacks):
                    foreground = img[:, i, ...].type(torch.uint8)
                    mask = call_back(foreground)
                    img[:, i, ...][foreground != mask] = 0
                    img[:, i, ...][mask.type(torch.bool)] = 1
            else:
                applied_img = img[:, self.applied_labels,
                                  ...].type(torch.uint8)
                foreground = torch.any(applied_img, dim=channel_dim)
                mask = self.filter_callbacks[0](foreground)
                background_mask = torch.unsqueeze(foreground != mask,
                                                  dim=channel_dim)
                background_mask = torch.repeat_interleave(
                    background_mask, len(self.applied_labels), dim=channel_dim)
                applied_img[background_mask] = 0
                img[:, self.applied_labels, ...] = applied_img.type(img.type())
            output = img

        return output
