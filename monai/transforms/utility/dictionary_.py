"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import Dict, Hashable, Mapping, Union, Sequence

import numpy as np
import torch

from monai.config import KeysCollection, DtypeLike
from monai.transforms.compose import MapTransform
from monai.transforms.utility.array_ import SelectChannel
from monai.transforms.utility.array import Transpose, CastToType
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.utils_ import print_tensor

class SelectChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """
    def __init__(self, keys: KeysCollection, channel_dim: int) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.selector = SelectChannel(channel_dim)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.selector(d[key])
        return d


class RemoveNegativeLabel():
    def __init__(self, keys: Sequence[str]) -> None:
        """
        Args:
            key: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype or torch.dtype,
                each element corresponds to a key in ``keys``.

        """
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key][d[key] < 0] = 0
        return d


class Transposed_():
    def __init__(self, keys: Sequence[str], axes: Sequence[int]) -> None:
        """
        Args:
            key: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of np.dtype or torch.dtype,
                each element corresponds to a key in ``keys``.

        """
        self.keys = keys
        self.axes = axes
        self.transpose = Transpose(axes)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transpose(d[key])
        return d


class SelectLabeld():
    def __init__(
        self,
        keys: Sequence[str],
        labels: Dict,
    ) -> None:
        """
        Args:
            key: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`

        """
        self.keys = keys
        self.labels = labels

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            new = np.zeros_like(d[key])
            for k, v in self.labels.items():
                new[d[key] == k] = v
            d[key] = new
        return d


class RideOnLabel():
    def __init__(self, keys, cat_dim = 1):
        """
        dict(type = 'RideOnLabel', keys = {'label': ('skeleton', ) } ),
        
        """
        self.keys = keys
        self.cat_dim = cat_dim
    
    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        
        for ok, nks in self.keys.items():
            nks = ensure_tuple(nks)
            # reserve the region specified by real gt mask 
            for ki in nks: 
                d[ki] = d[ki] * (d[ok] > 0)
                # print_tensor(f'confine values for {ki}', d[ki])
            cat_keys = (ok, ) + nks
            # print('keys to concat', cat_keys)
            d[ok] = torch.cat([d[k] for k in cat_keys], axis = self.cat_dim)
        return d



class CastToTyped_(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CastToType`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[Sequence[Union[DtypeLike, torch.dtype]], DtypeLike, torch.dtype] = np.float32,
        # allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            dtype: convert image to this data type, default is `np.float32`.
                it also can be a sequence of dtypes or torch.dtype,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys)
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.dtype = [np.float32 if 'float' == k else np.uint8 for k in self.dtype]
        self.converter = CastToType()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key, dtype in self.key_iterator(d, self.dtype):
            d[key] = self.converter(d[key], dtype=dtype)

        return d