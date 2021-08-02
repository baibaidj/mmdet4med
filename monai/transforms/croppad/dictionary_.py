"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
This file is copied from monai.transforms.croppad.array.py, support tensor input/output.
"""

from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
import pandas as pd

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform, Randomizable
from monai.transforms.croppad.array_ import (
    SpatialCrop_,
    CenterSpatialCrop_,
    SpatialMultiPyramidCrop_,
)
from monai.transforms.utils_ import (
    map_multilabel_to_indices,
    map_multilabel_bbox_to_indices,
    map_multilabel_convexhull_to_indices,
    generate_multipos_neg_label_crop_centers,
    MaskFgLocator3d, print_tensor
)
from monai.utils import fall_back_tuple, ensure_tuple_rep
import torch, os
import pickle
from monai.transforms.croppad.dictionary import map_binary_to_indices, SpatialCrop
from monai.transforms.utils_ import generate_pos_neg_label_crop_shift_centers
import pdb
from mmcv import Timer
import random


class SpatialCropd_(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialCrop_`.
    Either a spatial center and size must be provided, or alternatively if center and size
    are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        super().__init__(keys)
        self.cropper = SpatialCrop_(roi_center, roi_size, roi_start, roi_end)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.cropper(d[key])
        return d


class CenterSpatialCropd_(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.CenterSpatialCrop_`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: the size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, keys: KeysCollection, roi_size: Union[Sequence[int], int], jitter = 0, 
                      fixed_offset = (0, 0, 0)) -> None:
        super().__init__(keys)
        self.jitter = jitter
        # self.fixed_offset = fixed_offset
        self.cropper = CenterSpatialCrop_(roi_size, fixed_offset= fixed_offset)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        # rand_jitter = np.random.randint(-self.jitter, self.jitter) if self.jitter else 0
        for key in self.keys:
            j4c = d['img_meta_dict'].get('jitter4center', None)
            d[key], jitter_xyz = self.cropper(d[key], jitter = self.jitter if j4c is None else j4c)
            d['img_meta_dict']['jitter4center'] = jitter_xyz
        return d


from monai.transforms.croppad.array import SpatialPad, Method, NumpyPadMode
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]
class SpatialPadd_(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SpatialPad`.
    Performs padding to the data, symmetric for all sides or all on one side for each dimension.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        method: Union[Method, str] = Method.SYMMETRIC,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
        verbose = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            spatial_size: the spatial size of output data after padding.
                If its components have non-positive values, the corresponding size of input image will be used.
            method: {``"symmetric"``, ``"end"``}
                Pad image symmetric on every side or only pad at the end sides. Defaults to ``"symmetric"``.
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                It also can be a sequence of string, each element corresponds to a key in ``keys``.

        """
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padder = SpatialPad(spatial_size, method)
        self.verbose = verbose

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        # print('[Pad] data keys', d.keys())
        for key, m in zip(self.keys, self.mode):
            # try:
                # print_tensor(f'[Pad]Pre: {key}', d[key])
            d[key] = self.padder(d[key], mode=m)
            d['img_meta_dict'][f'padshape'] = d[key].shape[-3:]
            if self.verbose: print_tensor(f'[Pad]Post: {key}', d[key])
            # except ValueError:
            #     print(f'[Pad] tensor error fn is', d['img_meta_dict']['filename_or_obj'])
        return d

class RandCropShiftByPosNegLabeld(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop and shift random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        shift_percentage: a shift percentage value of image size as maximum offset.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        shift_percentage: float = 0.1,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.shift_percentage = shift_percentage

    def randomize(
        self,
        label: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        shift_percentage = np.random.uniform(-self.shift_percentage, self.shift_percentage, size=3)
        self.centers = generate_pos_neg_label_crop_shift_centers(
            self.spatial_size, shift_percentage, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> List[Dict[Hashable, np.ndarray]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.get(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.get(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise AssertionError
        if self.centers is None:
            raise AssertionError
        results: List[Dict[Hashable, np.ndarray]] = [{} for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
                    results[i][key] = cropper(img)
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class RandCropByPosNegMultiLabeld(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegMultiLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pyramid_scale: if use multiple pyramid level image outputs, set with multiple float type nums. e.g. [1.0, 2.0, 4.0]
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pyramid_scale: Union[Sequence[float], float] = (1.0, ),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        return_keys: KeysCollection = 'all',
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.pyramid_scale = pyramid_scale
        self.pyramid = True if len(pyramid_scale) > 1 else False
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.return_keys = return_keys

    def randomize(
        self,
        label: np.ndarray,
        fg_indices_list_: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices_list_ is None or bg_indices is None:
            fg_indices_list_, bg_indices_ = map_multilabel_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_list_ = fg_indices_list_
            bg_indices_ = bg_indices
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_list_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray, torch.Tensor]]) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = d.get(self.fg_indices_key, None) if self.fg_indices_key is not None else None
        bg_indices = d.get(self.bg_indices_key, None) if self.bg_indices_key is not None else None

        self.randomize(label, fg_indices, bg_indices, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]] = [dict() for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys and (self.return_keys == 'all' or key in self.return_keys):
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialMultiPyramidCrop_(roi_center=tuple(center), roi_size=self.spatial_size, pyramid_scale=self.pyramid_scale) if self.pyramid else SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
            elif self.return_keys == 'all' or key in self.return_keys:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results



class RandCropByDetectionBBoxd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByDetectionBBoxd`. with detection bbox info.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        return_keys: KeysCollection = 'all',
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.return_keys = return_keys

    def randomize(
        self,
        label: np.ndarray,
        fg_indices_list_: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices_list_ is None or bg_indices is None:
            fg_indices_list_, bg_indices_ = map_multilabel_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_list_ = fg_indices_list_
            bg_indices_ = bg_indices
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_list_, bg_indices_, self.R
        )

    def _correct_centers(
        self, center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
    ) -> List[np.ndarray]:
        for i, c in enumerate(center_ori):
            center_i = c
            for nn in range(3):
                if c[nn] < valid_start[nn]:
                    center_i[nn] = valid_start[nn]
                if c[nn] >= valid_end[nn]:
                    center_i[nn] = valid_end[nn] - 1
            center_ori[i] = center_i
        return center_ori

    def __call__(self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray, torch.Tensor]]) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        d = dict(data)
        bboxes = d['bbox']
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        self.centers = []
        for sample in range(self.num_samples):
            fg_indices_to_use = True if np.random.rand() < self.pos_ratio else False
            if fg_indices_to_use:
                bbox_nums = len(bboxes)
                bbox_idx = np.random.randint(bbox_nums)
                bbox = bboxes[bbox_idx]
                minx = bbox[0] - bbox[3]/2
                maxx = bbox[0] + bbox[3]/2
                miny = bbox[1] - bbox[4]/2
                maxy = bbox[1] + bbox[4]/2
                minz = bbox[2] - bbox[5]/2
                maxz = bbox[2] + bbox[5]/2
                cx = np.random.randint(minx, maxx)
                cy = np.random.randint(miny, maxy)
                cz = np.random.randint(minz, maxz)
            else:
                cx = np.random.randint(label.shape[1])
                cy = np.random.randint(label.shape[2])
                cz = np.random.randint(label.shape[3])
            self.centers.append([cx, cy, cz])

        valid_start = np.asarray(self.spatial_size)/2
        valid_end = np.asarray(label.shape[1:]) - valid_start
        self.centers = self._correct_centers(self.centers, valid_start, valid_end)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]] = [dict() for _ in range(self.num_samples)]
        fillbbox = False
        for key in data.keys():
            if key in self.keys and (self.return_keys == 'all' or key in self.return_keys):
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
                    roi_start = cropper.roi_start
                    new_bboxes = []
                    for bbox in bboxes:
                        minx = bbox[0] - bbox[3] / 2
                        maxx = bbox[0] + bbox[3] / 2
                        miny = bbox[1] - bbox[4] / 2
                        maxy = bbox[1] + bbox[4] / 2
                        minz = bbox[2] - bbox[5] / 2
                        maxz = bbox[2] + bbox[5] / 2
                        new_minx = minx - roi_start[0]
                        new_miny = miny - roi_start[1]
                        new_minz = minz - roi_start[2]
                        new_maxx = maxx - roi_start[0]
                        new_maxy = maxy - roi_start[1]
                        new_maxz = maxz - roi_start[2]
                        adp_minx = min(self.spatial_size[0], max(0, new_minx))
                        adp_miny = min(self.spatial_size[1], max(0, new_miny))
                        adp_minz = min(self.spatial_size[2], max(0, new_minz))
                        adp_maxx = min(self.spatial_size[0], max(0, new_maxx))
                        adp_maxy = min(self.spatial_size[1], max(0, new_maxy))
                        adp_maxz = min(self.spatial_size[2], max(0, new_maxz))

                        area1 = (new_maxx - new_minx) * (new_maxy - new_miny) * (new_maxz - new_minz)
                        area2 = (adp_maxx - adp_minx) * (adp_maxy - adp_miny) * (adp_maxz - adp_minz)
                        if area2 / area1 > 0.5 or (area2 / area1 > 0.25 and area2 > 1000):
                            new_bboxes.append([adp_minx, adp_miny, adp_maxx, adp_maxy, adp_minz, adp_maxz])
                    if fillbbox is False: # only fill once
                        results[i]['bbox'] = new_bboxes
                fillbbox = True

            elif key != 'bbox' and (self.return_keys == 'all' or key in self.return_keys):
                for i in range(self.num_samples):
                    results[i][key] = data[key]
        return results


class RandCropByLabelBBoxRegiond(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelBBoxRegion`.
    Crop random fixed sized regions with the each label minimum bounding box region
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pyramid_scale: if use multiple pyramid level image outputs, set with multiple float type nums. e.g. [1.0, 2.0, 4.0]
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pyramid_scale: Union[Sequence[float], float] = (1.0, ),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        whole_image_as_bg: bool = True,
        percentile_00_5: Optional[float] = None,
        return_keys: KeysCollection = 'all',
        custom_center_kargs = {},
        data_info_csv = '',
        num_classes = 1,
        verbose = False
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.pyramid_scale = pyramid_scale
        self.pyramid = True if len(pyramid_scale) > 1 else False
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.whole_image_as_bg = whole_image_as_bg
        self.percentile_00_5 = percentile_00_5
        self.return_keys = return_keys
        self.is_ribfrac = custom_center_kargs.pop('is_ribfrac', False)
        print(f'\n@@RandomCrop use ribfrac?  {self.is_ribfrac}')
        self.custom_center_kargs = custom_center_kargs
        self.info_tb = None
        self.verbose = verbose
        self.num_classes = num_classes
        if data_info_csv: 
            if os.path.exists(data_info_csv):
                self.info_tb = pd.read_csv(data_info_csv, index_col = 'cid')
                print(self.info_tb.head())
    def randomize(
        self,
        label: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> None:

        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        # with Timer(print_tmpl='\tRandCrop finding fg bg indices {:.3f} seconds'): #TODO
        fg_indices_list_, bg_indices_ = map_multilabel_bbox_to_indices(label, image, 
                                        whole_image_as_bg=self.whole_image_as_bg, 
                                        percentile_00_5=self.percentile_00_5, num_classes=self.num_classes)
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_list_, bg_indices_, self.R
        )
        if self.verbose: print('Initialize randomization got %d number of centers' %(len(self.centers)) )
    
    def custom_center_finding(self, label, base_value = None, jitter = 20):
        """
        this is built for hepatic vein
        label_mapping: 1 for hepatic vein, 2 for portal vein, 3 for artery, 4 for inferior vein, 5 = 1+4
        to obtain crops covering the hepatic vein, we use the centroid of region 5 as our center
        """

        if isinstance(base_value, int): base_value = [base_value]
        # assert isinstance(base_value, (tuple, list))
        if base_value is None:
            VeinLoc = MaskFgLocator3d(label[0], image_center=True)
        else:
            # print('center base value', base_value)
            self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
            mask_overlap_vein = (label == base_value[0]).astype(np.uint8)
            for bv in base_value[1:]:
                mask_overlap_vein += (label == bv).astype(np.uint8)
            VeinLoc = MaskFgLocator3d(mask_overlap_vein[0])

        center_dims = len(VeinLoc.obj_center_xyz)
        jitter_tuple = ensure_tuple_rep(jitter, center_dims)
        self.centers = [] # not use center
        for i, ix in enumerate(range(self.num_samples)):
            jitter_xyz = [0 if jitter ==0 else random.randrange(-jitter_tuple[d], jitter_tuple[d]) for d in range(center_dims)]
            # TODO: check if this jitter will be the same for different processes when using DDP
            center_jitter = [jitter_xyz[i] + VeinLoc.obj_center_xyz[i] for i in range(center_dims)]
            # print(f'[JITTER]', jitter_xyz, center_jitter)  #
            self.centers.append(center_jitter)
        
    def fetch_center_from_table(self, meta_data, jitter = 20):
        assert self.info_tb is not None
        # pdb.set_trace()
        fn = meta_data['filename_or_obj'].split(os.sep)[-1]
        case_name = '_'.join(fn.split('.')[0].split('_')[1:]) # [:2] or [1:]
        center_str = str(self.info_tb.loc[case_name, 'center_coord'])
        cstr = str(center_str).strip('"|[|]')
        ccord = [int(a) for a in cstr.split(',')]
        center_dims = len(ccord)
        jitter_tuple = ensure_tuple_rep(jitter, center_dims)
        self.centers = []
        for _ in enumerate(range(self.num_samples)):
            jitter_xyz = [random.randrange(-jitter_tuple[d], jitter_tuple[d]) for d in range(center_dims)]
            center_jitter = [jitter_xyz[i] + ccord[i] for i in range(center_dims)]
            self.centers.append(center_jitter)

    def fetch_center_from_table_rf(self, meta_data, jitter = 5):
        get_opposite_centroid_x = lambda c, shape_x: [ shape_x - a if i == 0 else a for i, a in enumerate(c)]

        assert self.info_tb is not None
        center_dims = 3
        jitter_tuple = ensure_tuple_rep(jitter, center_dims)
        # pdb.set_trace()
        fn = meta_data['filename_or_obj'].split(os.sep)[-1]
        case_name = fn.split(os.sep)[-1] # [:2] or [1:]

        # if self.verbose: print(f'Check, fn {fn}  casename {case_name}')

        centroids_str = str(self.info_tb.loc[case_name, 'centroids']).strip('"|[|]')
        shape_x = self.info_tb.loc[case_name, 'img_dim0']
        shape_z = self.info_tb.loc[case_name, 'img_dim2']
        fg_centroids = [[round(float(c)) for c in c3.split('-')] for c3 in centroids_str.split(',')]
        num_fgs = len(fg_centroids)
        # bg_opposite_centroids = [get_opposite_centroid_x(c3, shape_x) for c3 in fg_centroids]
        # if self.verbose: print(f'centroids {centroids_str} dim xz {shape_x} {shape_z}' )

        spine_rangexyz = [shape_x // 2 - 50, shape_x // 2 + 50] \
                        + [int(a) for a in self.info_tb.loc[case_name, 'spine_boundary'].split('-')][2:] \
                        + [self.spatial_size[2]//2 , shape_z - self.spatial_size[2]//2 ]
        spine_rangexyz = np.array(spine_rangexyz).reshape((3, 2))

        spine_randix_dim = lambda dim: random.randrange(spine_rangexyz[dim,0], spine_rangexyz[dim,1])
        coin_func = lambda : random.random() > 0.5
        # if self.verbose: print('spine range', spine_rangexyz)
        need_pos_num = min(int(self.num_samples * self.pos_ratio), num_fgs)
        need_opp_num = min(round((self.num_samples - need_pos_num)/2), num_fgs)
        sample2fg_ix = [random.randrange(0, num_fgs) for _ in range(need_pos_num)]
        sample2op_ix = [random.randrange(0, num_fgs) for _ in range(need_opp_num)]
        sample2spines = [[spine_randix_dim(i) for i in range(3)] 
                        for _ in range(min(self.num_samples - need_pos_num - need_opp_num, 2))]

        fg_opposite_centroids = []
        for c3 in fg_centroids:
            c3_opposite = [a for a in c3]
            for dim, cix in enumerate(c3):
                if dim == 0: c3_opposite[dim] = shape_x - cix if coin_func() else cix
                if dim == 1: c3_opposite[dim] = cix
                else: c3_opposite[dim] = spine_randix_dim(dim)
            fg_opposite_centroids.append(c3_opposite)

        self.centers = [fg_centroids[s] for s in sample2fg_ix] \
                        + [fg_opposite_centroids[s] for s in sample2op_ix] \
                        + sample2spines

        for i, this_cc in enumerate(self.centers):
            jitter_xyz = [random.randrange(-jitter_tuple[d], jitter_tuple[d]) for d in range(center_dims)]
            center_jitter = [jitter_xyz[i] + this_cc[i] for i in range(center_dims)]
            self.centers[i] = center_jitter

        if self.verbose: 
            print(f'[RandCrop] pos {need_pos_num}  pos_opposite {need_opp_num} spine {len(sample2spines)}')
            print('not jitter centers', self.centers)
            print('jitter centers', self.centers)


    def __call__(self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray, torch.Tensor]]) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        d = dict(data)
        image = d[self.keys[0]]
        label = d[self.label_key]
        # print('[RandCrop] data keys', d.keys())
        # with Timer(print_tmpl='\tRandCrop finding center {:.3f} seconds'): #TODO
        fetch_center_func = self.fetch_center_from_table_rf if self.is_ribfrac else self.fetch_center_from_table
        if self.info_tb is not None: fetch_center_func(d['img_meta_dict'], self.custom_center_kargs['jitter'])
        elif self.custom_center_kargs: self.custom_center_finding(label, **self.custom_center_kargs)
        else: self.randomize(label, image)

        self.num_centers = len(self.centers)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]] = [dict() for _ in range(self.num_centers)]
        for key in data.keys():
            if key in self.keys and (self.return_keys == 'all' or key in self.return_keys):
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialMultiPyramidCrop_(roi_center=tuple(center), 
                                                        roi_size=self.spatial_size, 
                                                        pyramid_scale=self.pyramid_scale) if self.pyramid else \
                            SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
                    # if self.verbose: print_tensor(f'\n {i} {key}raw center {center}', img)
                    # if self.verbose: print_tensor(f' {i} {key} crop ', results[i][key])
            elif self.return_keys == 'all' or key in self.return_keys: # for meta_dict
                for i in range(self.num_centers):
                    results[i][key] = d[key]
        if self.verbose: 
            print('Sampled data', results[0].keys())
        return results


class RandCropByLabelConvexRegiond(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelConvexRegion`.
    Crop random fixed sized regions with the each label convex hull region
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pyramid_scale: if use multiple pyramid level image outputs, set with multiple float type nums. e.g. [1.0, 2.0, 4.0]
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        return_keys: return the keys in dict. only

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pyramid_scale: Union[Sequence[float], float] = (1.0, ),
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        whole_image_as_bg: bool = True,
        percentile_00_5: Optional[float] = None,
        return_keys: KeysCollection = 'all',
    ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        self.pyramid_scale = pyramid_scale
        self.pyramid = True if len(pyramid_scale) > 1 else False
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.whole_image_as_bg = whole_image_as_bg
        self.percentile_00_5 = percentile_00_5
        self.return_keys = return_keys

    def randomize(
        self,
        convexhull: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=convexhull.shape[1:])

        fg_indices_list_, bg_indices_ = map_multilabel_convexhull_to_indices(convexhull, image, whole_image_as_bg=self.whole_image_as_bg, percentile_00_5=self.percentile_00_5)
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, convexhull.shape[1:], fg_indices_list_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray, torch.Tensor]]) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        d = dict(data)
        image = d[self.keys[0]]
        convexhull = d[self.label_key]

        self.randomize(convexhull, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results: List[Dict[Hashable, np.ndarray]] = [dict() for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys and (self.return_keys == 'all' or key in self.return_keys):
                img = d[key]
                for i, center in enumerate(self.centers):
                    cropper = SpatialMultiPyramidCrop_(roi_center=tuple(center), roi_size=self.spatial_size, pyramid_scale=self.pyramid_scale) if self.pyramid else SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
            elif self.return_keys == 'all' or key in self.return_keys:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class RandCropBySkeletiond(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropBySkeletiond`.
    Crop random fixed sized regions with the each label convex hull region
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        skeletion_key: name of key for current case, this will be used for finding current skeletion.
        skeletion_file: file name of the skeletions.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pyramid_scale: Union[Sequence[float], float] = 1.0,
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        class_weight: class weight.
        return_keys: return the keys in dict. only

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """
    def __init__(self,
                 keys: KeysCollection,
                 skeletion_key: str,
                 skeletion_file: str,
                 spatial_size: Union[Sequence[int], int],
                pyramid_scale: Union[Sequence[float], float] = (1.0, ),
                 num_samples: int = 1,
                 pos: float = 1,
                 neg: float = 1,
                 whole_image_as_bg: bool = True,
                 class_weight=None) -> None:

        super().__init__(keys)
        self.skeletion_key = skeletion_key
        self.skeletion_file = skeletion_file
        self.spatial_size: Union[Tuple[int, ...], Sequence[int],
                                 int] = spatial_size
        self.pyramid_scale = pyramid_scale
        self.pyramid = True if len(pyramid_scale) > 1 else False
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.whole_image_as_bg = whole_image_as_bg
        self.class_weight = class_weight
        self.initialize()

    def initialize(self):
        with open(self.skeletion_file, 'rb') as f:
            data = pickle.load(f)
        for case in data.values():
            for organ, tree in case['label'].items():
                tree['vertice'] = np.array(tree['vertice'])
                tree['path'] = [
                    np.array(t, dtype=np.uint32) for t in tree['path']
                ]
                if 'radius' in tree:
                    tree['radius'] = np.array(tree['radius'], dtype=np.uint32)
                if isinstance(organ, str):
                    case['label'][int(organ)] = tree
                    del case['label'][organ]
        self.skeletion_data = data

        if self.class_weight is None:
            self.class_weight = [1] * len(case['label'].keys())
        else:
            assert len(self.class_weight) == len(
                case['label'].keys()
            ), 'class weight length is not equal to the class found in skeletion file.'
        self.class_weight = self.class_weight / np.sum(self.class_weight)

    def randomize(
        self,
        key: str,
        image: Optional[np.ndarray] = None,
    ) -> None:
        case = self.skeletion_data[key]['label']
        self.centers = []
        pos_flag = np.random.rand(self.num_samples) < self.pos_ratio
        for flag in pos_flag:
            if flag:
                organ = np.random.choice(sorted(list(case.keys())),
                                         p=self.class_weight)
                tree = case[organ]
                idx = np.random.choice(tree['vertice'].shape[0])
                center = tree['vertice'][idx]
            else:
                center = np.array([
                    np.random.randint(p // 2, s - p // 2)
                    for p, s in zip(self.spatial_size, image.shape[1:])
                ])
            self.centers.append(center)

    def __call__(
        self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray,
                                            torch.Tensor, str]]
    ) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        image = data[self.keys[0]]
        skeletion_key = data[self.skeletion_key]

        self.randomize(skeletion_key, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results = []
        rest_keys = set(data.keys()) - set(self.keys)
        for i in range(self.num_samples):
            shot = {}
            center = self.centers[i]
            cropper = SpatialMultiPyramidCrop_(roi_center=tuple(center), roi_size=self.spatial_size, pyramid_scale=self.pyramid_scale) if self.pyramid else SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
            for key in self.keys:
                shot[key] = cropper(data[key])
            for key in rest_keys:
                shot[key] = data[key]
            results.append(shot)
        return results


def calc_slice_and_pad(center, src_shape, dst_shape):
    """calculate the slice to generate the patch and pad size.

    Args:
        center ([tuple]): patch center
        src_shape ([tuple]): image shape 
        dst_shape ([tuple]): patch size.

    Returns:
        [(tuple(slice), tuple(int))]: the patch slice and pad size. 
    """
    bbox = [(c - s // 2, c + s // 2) for c, s in zip(center, dst_shape)]
    slices = [
        slice(0 if l < 0 else l, s if u > s else u)
        for (l, u), s in zip(bbox, src_shape)
    ]
    pads = [(abs(l) if l < 0 else 0, u - s if u > s else 0)
            for (l, u), s in zip(bbox, src_shape)]
    return tuple(slices), pads


class RandCropByLabelBBoxd(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByLabelBBoxd`.
    Crop random fixed sized regions with the each label convex hull region
    based on the Pos Neg Ratio. This is used for multi-label case. Each label will be cropped with same probability.
    And will return a list of dictionaries for all the cropped images.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        bbox_key: name of key for current case, this will be used for finding current skeletion.
        bbox_file: file name of the skeletions.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `data[label_key]` will be used.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        class_weight: class weight.
        image_enclose_patch: whether the cropped patch is completely enclosed in the image.
        return_keys: return the keys in dict. only

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """
    def __init__(self,
                 keys: KeysCollection,
                 bbox_key: str,
                 bbox_file: str,
                 spatial_size: Union[Sequence[int], int],
                 num_samples: int = 1,
                 pos: float = 1,
                 neg: float = 1,
                 whole_image_as_bg: bool = True,
                 class_weight=None,
                 image_enclose_patch=True) -> None:
        super().__init__(keys)
        self.bbox_key = bbox_key
        self.bbox_file = bbox_file
        self.spatial_size: Union[Tuple[int, ...], Sequence[int],
                                 int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.whole_image_as_bg = whole_image_as_bg
        self.class_weight = class_weight
        self.image_enclose_patch = image_enclose_patch
        self.initialize()

    def initialize(self):
        with open(self.bbox_file, 'rb') as f:
            data = pickle.load(f)
        for case in data.values():
            for organ, bbox in case.items():
                if isinstance(organ, str):
                    case[int(organ)] = bbox
                    del case[organ]
        self.bbox_data = data

        if self.class_weight is None:
            self.class_weight = [1] * len(case.keys())
        else:
            assert len(self.class_weight) == len(
                case.keys()
            ), 'class weight length is not equal to the class found in skeletion file.'
        self.class_weight = self.class_weight / np.sum(self.class_weight)

    def randomize(
        self,
        key: str,
        image: Optional[np.ndarray] = None,
    ) -> None:
        case = self.bbox_data[key]
        self.centers = []
        pos_flag = np.random.rand(self.num_samples) < self.pos_ratio
        for flag in pos_flag:
            if flag:
                organ = np.random.choice(sorted(list(case.keys())),
                                         p=self.class_weight)
                bbox = case[organ]
                if self.image_enclose_patch:
                    bbox = [
                        (max(p // 2, l), min(s - p // 2, h)) for s, p, (
                            l,
                            h) in zip(image.shape[1:], self.spatial_size, bbox)
                    ]
                bbox = [(l, h) if l < h else (ol, oh)
                        for (l, h), (ol, oh) in zip(bbox, case[organ])]
                center = [np.random.randint(l, h + 1) for l, h in bbox]

            else:
                bbox = [(p // 2, s - p // 2)
                        for p, s in zip(self.spatial_size, image.shape[1:])]
                bbox = [(l, h) if l < h else (0, s - 1)
                        for (l, h), s in zip(bbox, image.shape[1:])]
                center = [np.random.randint(l, h + 1) for l, h in bbox]
            self.centers.append(center)

    def __call__(
        self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray,
                                            torch.Tensor, str]]
    ) -> List[Dict[Hashable, Union[np.ndarray, torch.Tensor]]]:
        image = data[self.keys[0]]
        bbox_key = data[self.bbox_key]

        self.randomize(bbox_key, image)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results = []
        rest_keys = set(data.keys()) - set(self.keys)
        for i in range(self.num_samples):
            shot = {}
            center = self.centers[i]
            slices, pads = calc_slice_and_pad(center, image.shape[1:],
                                              self.spatial_size)
            for key in self.keys:
                arr = np.array([c[slices] for c in data[key]])
                arr = np.pad(arr, [(0, 0)] + pads,
                             constant_values=data[key].min())
                assert arr.shape[1:] == self.spatial_size, (
                    f'bbox: {bbox_key}, '
                    f'real shape: {arr.shape[1:]}, '
                    f'required shape: {self.spatial_size}, '
                    f'image shape: {image.shape[1:]}')
                shot[key] = arr
            for key in rest_keys:
                shot[key] = data[key]
            results.append(shot)
        return results

class ConvertLabeld(MapTransform):

    def __init__(
        self,
        keys = ['label'],
        label_mapping = None,
        value4outlier = 0,
        ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        super().__init__(keys)
        self.label_mapping = label_mapping
        self.value4outlier = value4outlier

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], (np.ndarray, torch.Tensor)):
                d[key] = convert_label(d[key], self.label_mapping, value4outlier= self.value4outlier)
        return d


def convert_label(label, label_mapping = None, inverse=False, value4outlier = 0):
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
        label[label > max_value] = value4outlier
    return label


class CropByRoid:
    def __init__(self,
                 keys,
                 roi_key,
                 use_labels,
                 size=(256, 256, 128),
                 margin=32):
        self.keys = keys
        self.roi_key = roi_key
        self.use_labels = use_labels
        self.minimum_size = size
        self.margin = margin

    def __call__(
        self, data: Mapping[Hashable, Union[List[np.ndarray], np.ndarray,
                                            torch.Tensor, str]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        prev_seg = data[self.roi_key].copy()
        if prev_seg.shape[-1] == 1:
            prev_seg = np.squeeze(prev_seg, -1)
        for k in self.keys:
            assert prev_seg.shape[1:] == data[k].shape[
                1:], f'different shape: {prev_seg.shape}, {data[k].shape}'

        prev_seg = prev_seg[0]
        # roi mask
        labels = [k for k in self.use_labels.keys() if k != 0]
        foreground = np.in1d(prev_seg,
                             labels).reshape(prev_seg.shape).astype(np.uint8)
        bbox = ndimage.find_objects(foreground)[0]

        # add margin
        bbox = [
            slice(b.start - self.margin // 2, b.stop + self.margin // 2)
            for b in bbox
        ]

        # extend to minimum size
        bbox = [
            b if (b.stop - b.start) > sz else slice(
                b.start - (sz - (b.stop - b.start)) // 2, b.stop +
                (sz - (b.stop - b.start)) // 2)
            for b, sz in zip(bbox, self.minimum_size)
        ]

        # don't extend out of the data
        # calculate pad to restore to original shape in the furture.
        pad = []
        slices = []
        for box, sp in zip(bbox, prev_seg.shape):
            start = box.start
            stop = box.stop
            # keep in the origin shape
            start = 0 if start < 0 else start
            stop = sp if stop > sp else stop
            slc = slice(start, stop)
            slices.append(slc)
            pad.append((slc.start, sp - slc.stop))

        for k in self.keys:
            data[k] = data[k][tuple([slice(0, data[k].shape[0])] + slices)]
        data['pad_roi'] = pad
        return data
