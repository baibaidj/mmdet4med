"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
This file is copied from monai.transforms.croppad.array.py, support tensor input/output.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.utils_ import (
    map_multilabel_to_indices,
    map_multilabel_bbox_to_indices,
    map_multilabel_convexhull_to_indices,
    generate_multipos_neg_label_crop_centers,
)
from monai.utils import ensure_tuple, fall_back_tuple
from monai.utils.utils_ import resize
import torch
import random


class SpatialCrop_(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """
        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            self.roi_end = np.maximum(self.roi_start + roi_size, self.roi_start)
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.roi_start), len(self.roi_end), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start[:sd], self.roi_end[:sd])]
        return img[tuple(slices)]


class SpatialMultiPyramidCrop_(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        pyramid_scale: Union[Sequence[float], float] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            pyramid_scale: pyramid scale list for different scaled patch, e.g [1.0, 2.0, 4.0]
        """
        if roi_center is not None and roi_size is not None and pyramid_scale is not None:
            self.pyramid_levels = len(pyramid_scale)
            self.roi_size = roi_size
            self.pyramid_scale = pyramid_scale
            roi_center = np.asarray(roi_center, dtype=np.int16)
            self.roi_start_list, self.roi_end_list = list(), list()
            # correct roi_center
            roi_start = np.maximum(roi_center - np.floor_divide(roi_size, 2), 0)
            roi_end = np.maximum(roi_start + roi_size, roi_start)
            roi_center = roi_start + (roi_end - roi_start)/2
            roi_start = np.round(roi_start).astype(np.int16)
            roi_end = np.round(roi_end).astype(np.int16)
            self.roi_start_list.append(roi_start)
            self.roi_end_list.append(roi_end)

            for p in range(1, self.pyramid_levels):
                roi_size_ = np.asarray(roi_size, dtype=np.int16)
                roi_size_ = roi_size_ * pyramid_scale[p]
                roi_size_ = np.round(roi_size_).astype(np.uint16)
                roi_start = roi_center - np.floor_divide(roi_size_, 2)
                roi_end = roi_start + roi_size_
                roi_start = np.round(roi_start).astype(np.int16)
                roi_end = np.round(roi_end).astype(np.int16)
                self.roi_start_list.append(roi_start)
                self.roi_end_list.append(roi_end)
        else:
            raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        # pading image
        img_shape = np.squeeze(np.asarray(img.shape[1:]))
        diff_start = np.zeros(self.roi_start_list[0].shape, dtype=np.int16)
        diff_end = np.zeros(self.roi_start_list[0].shape, dtype=np.int16)
        for p in range(self.pyramid_levels):
            diff_start = np.maximum(np.maximum((0 - self.roi_start_list[p]), 0), diff_start)
            diff_end = np.maximum(np.maximum((self.roi_end_list[p] - img_shape), 0), diff_end)

        diff = np.stack((np.insert(diff_start, 0, 0), np.insert(diff_end, 0, 0)))
        diff = np.transpose(diff)
        img = np.pad(img, diff, 'constant', constant_values=0)
        img_pyramids = list()
        for p in range(self.pyramid_levels):
            self.roi_start_list[p] += diff_start
            self.roi_end_list[p] += diff_start
            sd = min(len(self.roi_start_list[p]), len(self.roi_end_list[p]), len(img.shape[1:]))  # spatial dims
            slices = [slice(None)] + [slice(s, e) for s, e in zip(self.roi_start_list[p][:sd], self.roi_end_list[p][:sd])]
            img_p = img[tuple(slices)]
            if self.pyramid_scale[p] != 1.0:
                img_p = resize(img_p, self.roi_size, mode='trilinear', align_corners=False, GPU=False)
            img_pyramids.append(img_p)
        img_pyramids = np.asarray(img_pyramids)
        img_pyramids = np.squeeze(img_pyramids, axis=1)
        return img_pyramids

class RandCropByPosNegMultiLabel(Randomizable, Transform):
    """
    Crop random fixed sized regions with the center being a multiple foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices_list = fg_indices_list
        self.bg_indices = bg_indices

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

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices_list: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if image is None:
            image = self.image
        if fg_indices_list is None or bg_indices is None:
            if self.fg_indices_list is not None and self.bg_indices is not None:
                fg_indices_list = self.fg_indices_list
                bg_indices = self.bg_indices
            else:
                fg_indices_list, bg_indices = map_multilabel_to_indices(label, image, self.image_threshold)
        self.randomize(label, fg_indices_list, bg_indices, image)
        results: List[np.ndarray] = list()
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results


class RandCropByLabelBBoxRegion(Randomizable, Transform):
    """
    Crop random fixed sized regions with the each label minimum bounding box region
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[0, 0, 0],
          [0, 1, 0, 0, 0],     -->     [0, 1, 0],      [0, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices_list = fg_indices_list
        self.bg_indices = bg_indices

    def randomize(
        self,
        label: np.ndarray,
        fg_indices_list_: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices_list_ is None or bg_indices is None:
            fg_indices_list_, bg_indices_ = map_multilabel_bbox_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_list_ = fg_indices_list_
            bg_indices_ = bg_indices
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_list_, bg_indices_, self.R
        )

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices_list: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if image is None:
            image = self.image
        if fg_indices_list is None or bg_indices is None:
            if self.fg_indices_list is not None and self.bg_indices is not None:
                fg_indices_list = self.fg_indices_list
                bg_indices = self.bg_indices
            else:
                fg_indices_list, bg_indices = map_multilabel_bbox_to_indices(label, image, self.image_threshold)
        self.randomize(label, fg_indices_list, bg_indices, image)
        results: List[np.ndarray] = list()
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results


class RandCropByLabelConvexRegion(Randomizable, Transform):
    """
    Crop random fixed sized regions with the each label convex hull region
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            If its components have non-positive values, the corresponding size of `label` will be used.
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[np.ndarray] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[np.ndarray] = None,
        image_threshold: float = 0.0,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.fg_indices_list = fg_indices_list
        self.bg_indices = bg_indices

    def randomize(
        self,
        label: np.ndarray,
        fg_indices_list_: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices_list_ is None or bg_indices is None:
            fg_indices_list_, bg_indices_ = map_multilabel_convexhull_to_indices(label, image, self.image_threshold)
        else:
            fg_indices_list_ = fg_indices_list_
            bg_indices_ = bg_indices
        self.centers = generate_multipos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[1:], fg_indices_list_, bg_indices_, self.R
        )

    def __call__(
        self,
        img: np.ndarray,
        label: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        fg_indices_list: Optional[List[np.ndarray]] = None,
        bg_indices: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices_list: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if image is None:
            image = self.image
        if fg_indices_list is None or bg_indices is None:
            if self.fg_indices_list is not None and self.bg_indices is not None:
                fg_indices_list = self.fg_indices_list
                bg_indices = self.bg_indices
            else:
                fg_indices_list, bg_indices = map_multilabel_convexhull_to_indices(label, image, self.image_threshold)
        self.randomize(label, fg_indices_list, bg_indices, image)
        results: List[np.ndarray] = list()
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop_(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results



class SpatialCropDJ(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    It can support to crop ND spatial (channel-first) data.
    Either a spatial center and size must be provided, or alternatively,
    if center and size are not provided, the start and end coordinates of the ROI must be provided.
    """

    def __init__(
        self,
        roi_center: Optional[Sequence[int]] = None,
        roi_size: Optional[Sequence[int]] = None,
        roi_start: Optional[Sequence[int]] = None,
        roi_end: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI.
        """

        if roi_center is not None and roi_size is not None:
            roi_center = np.asarray(roi_center, dtype=np.int16)
            roi_size = np.asarray(roi_size, dtype=np.int16)
            self.roi_start = roi_center - np.floor_divide(roi_size, 2)
            self.roi_end = self.roi_start + roi_size
            self.roi_size = roi_size
        else:
            if roi_start is None or roi_end is None:
                raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
            self.roi_start = np.maximum(np.asarray(roi_start, dtype=np.int16), 0)
            self.roi_end = np.maximum(np.asarray(roi_end, dtype=np.int16), self.roi_start)
            self.roi_size = self.roi_end - self.roi_start


    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        # print('img', img.shape)
        image_shape = np.array(img.shape[1:])
        end_xyz_delta = self.roi_end - image_shape
        patch_shape = (img.shape[0], ) + tuple(self.roi_size)
        patch_image = np.full(patch_shape, img.min(), dtype = img.dtype)
        start_xyz_img = self.roi_start.clip(0) 
        end_xyz_img = self.roi_end.clip(0, image_shape)
        # slice4img = [slice(start_xyz_img[i], end_xyz_img[i]) for i in range(3)]

        start_xyz_patch = np.where(self.roi_start < 0, np.abs(self.roi_start),0)
        end_xyz_patch = np.where(end_xyz_delta > 0, self.roi_size - np.abs(end_xyz_delta), self.roi_size) #
        # slice4patch = [slice(start_xyz_patch[i], end_xyz_patch[i]) for i in range(3)]

        sd = min(len(self.roi_start), len(self.roi_end), len(img.shape[1:]))  # spatial dims
        slices4img = tuple([slice(None)] + [slice(s, e) for s, e in zip(start_xyz_img[:sd], end_xyz_img[:sd])])
        slices4patch = tuple([slice(None)] + [slice(s, e) for s, e in zip(start_xyz_patch[:sd], end_xyz_patch[:sd])])

        patch_image[slices4patch] = img[slices4img]

        return patch_image, slices4img, slices4patch


randint = lambda x: random.randrange(-x, x)
class CenterSpatialCrop_j(Transform):
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            If its components have non-positive values, the corresponding size of input image will be used.
    """

    def __init__(self, roi_size: Union[Sequence[int], int], fixed_offset = (0, 0, 0)) -> None:
        self.roi_size = roi_size
        self.fixed_offset = fixed_offset

    def __call__(self, img: Union[np.ndarray, torch.Tensor], jitter_xyz = (0, 0, 0)) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        self.roi_size = fall_back_tuple(self.roi_size, img.shape[1:])
        # jitter_xyz = [randint(jitter) if jitter else 0 for _ in range(3)]
        img_shape = list(img.shape[1:])
        center = [min(s - self.roi_size[i]//2,
                    max(s // 2 + jitter_xyz[i] + self.fixed_offset[i], self.roi_size[i]//2)
                    )
                 for i, s in enumerate(img_shape)]
        cropper = SpatialCropDJ(roi_center=center, roi_size=self.roi_size)
        patch_img, slice4img, slice4patch = cropper(img)
        return patch_img, slice4img, slice4patch