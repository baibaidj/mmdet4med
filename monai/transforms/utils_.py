"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

import random
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import IndexSelection
from monai.utils import ensure_tuple, ensure_tuple_rep, ensure_tuple_size, fall_back_tuple, min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)
print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())


def generate_pos_neg_label_crop_shift_centers(
    spatial_size: Union[Sequence[int], int],
    shift_percentage: Sequence[float],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: np.ndarray,
    bg_indices: np.ndarray,
    rand_state: np.random.RandomState = np.random,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        shift_percentage: shift percentage of image size.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The proposed roi is larger than the image.")

    shift_x = spatial_size[0] * shift_percentage[0]
    shift_y = spatial_size[1] * shift_percentage[1]
    shift_z = spatial_size[2] * shift_percentage[2]

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
        if valid_start[i] == valid_end[i]:
            valid_end[i] += 1

    def _correct_centers(
        center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
    ) -> List[np.ndarray]:
        for i, c in enumerate(center_ori):
            center_i = c
            if c < valid_start[i]:
                center_i = valid_start[i]
            if c >= valid_end[i]:
                center_i = valid_end[i] - 1
            center_ori[i] = center_i
        return center_ori

    centers = []
    fg_indices, bg_indices = np.asarray(fg_indices), np.asarray(bg_indices)
    if fg_indices.size == 0 and bg_indices.size == 0:
        raise ValueError("No sampling location available.")

    if fg_indices.size == 0 or bg_indices.size == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        center_ori[0] = round(center_ori[0] + shift_x)
        center_ori[1] = round(center_ori[1] + shift_y)
        center_ori[2] = round(center_ori[2] + shift_z)
        centers.append(_correct_centers(center_ori, valid_start, valid_end))

    return centers


def generate_multipos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices_list: List[np.ndarray],
    bg_indices: np.ndarray,
    rand_state: np.random.RandomState = np.random,
    verbose = False,
) -> List[List[np.ndarray]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio from multiple foreground indices list
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]

    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices_list: a list of pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.

    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.

    """
    spatial_size = fall_back_tuple(spatial_size, default=label_spatial_shape)
    if not (np.subtract(label_spatial_shape, spatial_size) >= 0).all():
        raise ValueError("The proposed roi is larger than the image.")

    # Select subregion to assure valid roi
    valid_start = np.floor_divide(spatial_size, 2)
    # add 1 for random
    valid_end = np.subtract(label_spatial_shape + np.array(1), spatial_size / np.array(2)).astype(np.uint16)
    # int generation to have full range on upper side, but subtract unfloored size/2 to prevent rounded range
    # from being too high
    for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
        if valid_start[i] == valid_end[i]:
            valid_end[i] += 1

    def _correct_centers(
        center_ori: List[np.ndarray], valid_start: np.ndarray, valid_end: np.ndarray
    ) -> List[np.ndarray]:
        for i, c in enumerate(center_ori):
            center_i = c
            if c < valid_start[i]:
                center_i = valid_start[i]
            if c >= valid_end[i]:
                center_i = valid_end[i] - 1
            center_ori[i] = center_i
        return center_ori

    centers = []

    if not len(fg_indices_list) or not len(bg_indices):
        if not len(fg_indices_list) and not len(bg_indices):
            raise ValueError("No sampling location available.")
        warnings.warn(
            f"N foreground {len(fg_indices_list)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if not len(fg_indices_list) else 1
    if verbose: print('\n')
    for s in range(num_samples):
        fg_indices = None if not len(fg_indices_list) else fg_indices_list[rand_state.randint(len(fg_indices_list))]
        use_fg_samples = rand_state.rand() < pos_ratio
        indices_to_use = fg_indices if use_fg_samples else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        center = np.unravel_index(indices_to_use[random_int], label_spatial_shape)
        if verbose: print(f'[ix{s}] pos ratio {pos_ratio} sample regions Foreground? {use_fg_samples}', center)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(_correct_centers(center_ori, valid_start, valid_end))

    return centers


def map_multilabel_to_indices(
    label: np.ndarray,
    image: Optional[np.ndarray] = None,
    image_threshold: float = 0.0,
    whole_image_as_bg: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute the multiple foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 2], [2, 0, 1], [1, 1, 0]]])``
    ``foreground indices list = [np.array([1, 5, 6, 7]), np.array([2, 3])]`` and ``background indices = np.array([0, 4, 8])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.

    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    # max_label = int(np.max(label))
    label_values = np.unique(label)
    label_values = np.delete(label_values, 0)
    fg_indices_list = []
    for fgv in label_values: # range(1, max_label + 1):
        label_flat_ = np.any(label == fgv, axis=0).ravel()  # in case label has multiple dimensions
        fg_indices = np.nonzero(label_flat_)[0]
        fg_indices_list.append(fg_indices)

    if whole_image_as_bg or len(fg_indices_list) < 1:
        bg_indices = np.arange(label.size)
    else:
        label_flat = np.hstack(fg_indices_list)
        bg_indices = ~label_flat

    if image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, bg_indices))[0]
    else:
        bg_indices = np.nonzero(bg_indices)[0]
    return fg_indices_list, bg_indices


def map_multilabel_bbox_to_indices(
    label: np.ndarray,
    image: np.ndarray,
    image_threshold: float = 0.0,
    whole_image_as_bg: bool = True,
    percentile_00_5: Optional[float] = None,
    num_classes = 2,
    verbose = False
):
    """
    Compute the minimum boundingbox of foreground and background of input label data, return the indices after fattening.
    For example:
    ``label = np.array([[[0, 1, 1], [0, 0, 1], [0, 1, 0]]])``
    ``foreground indices = np.array([1, 2, 4, 5, 7, 8])`` and ``background indices = np.array([0, 3, 6])``

    Args:
        label: use the label data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.

    """
    # Prepare fg/bg indices
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel

    fg_indices_list = []
    for fgv in range(1, max(2, num_classes)):
    # label_bbox = set_bbx_indices(label, np.where(label > 0))
        label_flat = np.any(label == fgv, axis=0).ravel()  # in case label has multiple dimensions
        fg_indices = np.nonzero(label_flat)[0]
        if verbose: print('\nfinding Foreground indices', len(fg_indices), fg_indices[:3])
        fg_indices_list.append(fg_indices)

    label_flat = np.hstack(fg_indices_list)
    if whole_image_as_bg and percentile_00_5 is None:
        bg_indices = np.arange(label.size)
    elif whole_image_as_bg and percentile_00_5 is not None:
        whole_bg_bbox = np.logical_and(image > percentile_00_5, label == 0)
        bg_flat = np.any(whole_bg_bbox > 0, axis=0).ravel()
        bg_indices = np.nonzero(bg_flat)[0]
    elif whole_image_as_bg is False and image is not None:
        img_flat = np.any(image > image_threshold, axis=0).ravel()
        bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
    elif whole_image_as_bg is False and image is None:
        bg_indices = np.nonzero(~label_flat)[0]
    else:
        raise ValueError(f"input params whole_image_as_bg, percentile_00_5 are err with {whole_image_as_bg, percentile_00_5}.")
    if verbose: print('\tfinding Background indices', len(bg_indices), bg_indices[:3])
    return fg_indices_list, bg_indices


def map_multilabel_convexhull_to_indices(
    convexhull: np.ndarray,
    image: np.ndarray,
    image_threshold: float = 0.0,
    whole_image_as_bg: bool = True,
    percentile_00_5: Optional[float] = None,
):
    """
    Compute the foreground and background convex hull of input label data, return the indices after fattening.

    Args:
        convexhull: use the convexhull data to get the foreground/background information.
        image: if image is not None, use ``label = 0 & image > image_threshold``
            to define background. so the output items will not map to all the voxels in the label.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.

    """
    # Prepare fg/bg indices
    max_label_size = convexhull.shape[0]
    fg_indices_list = []
    for fgv in range(1, max_label_size):
        label = convexhull[fgv]
        label = label[np.newaxis, :]
        label_flat = np.any(label > 0, axis=0).ravel()  # in case label has multiple dimensions
        fg_indices = np.nonzero(label_flat)[0]
        fg_indices_list.append(fg_indices)

    if whole_image_as_bg and percentile_00_5 is None:
        bg_indices = np.arange(image.size)
    elif whole_image_as_bg and percentile_00_5 is not None:
        whole_bg_convex = convexhull[0]
        whole_bg_convex = whole_bg_convex[np.newaxis, :]
        bg_flat = np.any(whole_bg_convex > 0, axis=0).ravel()
        bg_indices = np.nonzero(bg_flat)[0]
    else:
        label_bg = convexhull[0]
        label_bg = label_bg[np.newaxis, :]
        for i in range(1, max_label_size):
            c = convexhull[i]
            c = c[np.newaxis, :]
            label_bg[c > 0] = 0
        label_bg_flat_ = np.any(label_bg > 0, axis=0).ravel()  # in case label has multiple dimensions
        if image is not None:
            img_flat = np.any(image > image_threshold, axis=0).ravel()
            bg_indices = np.nonzero(np.logical_and(img_flat, label_bg_flat_))[0]
        else:
            bg_indices = np.nonzero(label_bg_flat_)[0]
    return fg_indices_list, bg_indices


def get_topn_connected_component_mask(img: torch.Tensor,
                                      connectivity: Optional[int] = None,
                                      topn=1,
                                      to_largest_ratio=0) -> torch.Tensor:
    """
    Gets the top n largest connected component mask of an image.

    Args:
        img: Image to get largest connected component from. Shape is (batch_size, spatial_dim1 [, spatial_dim2, ...])
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    img_arr = img.detach().cpu().numpy()
    largest_cc = np.zeros(shape=img_arr.shape, dtype=img_arr.dtype)
    for i, item in enumerate(img_arr):
        item = measure.label(item, connectivity=connectivity)
        if item.max() != 0:
            bincount = np.bincount(item.flat)
            count_arg = np.argsort(bincount[1:])[::-1] + 1
            count_arg = count_arg[:topn]
            for arg in count_arg:
                if bincount[arg] / bincount[count_arg[0]] < to_largest_ratio:
                    break
                largest_cc[i][item == arg] = 1
    return torch.as_tensor(largest_cc, device=img.device)


def threshold_connected_component_mask(
    img: Union[torch.Tensor, np.ndarray],
    threshold: int,
    connectivity: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Remove component if its volume is small than threshold.

    Args:
        img: The input mask. Shape is (batch_size, spatial_dim1 [, spatial_dim2, ...])
        threshold: The threshold volume, the component will be remove if its volume is small than threshold.
        connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
            Accepted values are ranging from  1 to input.ndim. If ``None``, a full
            connectivity of ``input.ndim`` is used.
    """
    if isinstance(img, torch.Tensor):
        img_arr = img.detach().cpu().numpy()
    img_arr = img_arr.copy()
    for i, item in enumerate(img_arr):
        item = measure.label(item, connectivity=connectivity)
        if item.max() != 0:
            bincount = np.bincount(item.flat)
            for arg, vol in enumerate(bincount):
                if vol > threshold:
                    continue
                img_arr[i][item == arg] = 0
    if isinstance(img, torch.Tensor):
        img_arr = torch.as_tensor(img_arr, device=img.device)
    return img_arr




class MaskFgLocator3d():

    def __init__(self, mask_3d, image_center = False):
        self.mask_3d = mask_3d
        self.dim = len(mask_3d.shape)
        self.image_center = image_center
        self.pos_idx_dim()

    @property
    def obj_loc_property(self):
        fg_coords = np.where(self.mask_3d > 0)
        
        self.pos_x_ixs = list(fg_coords[0]) #[z for z in range(self.mask_3d.shape[0]) if self.mask_3d[z].max() > 0]
        self.pos_y_ixs = list(fg_coords[1]) #[z for z in range(self.mask_3d.shape[1]) if self.mask_3d[:, z, ...].max() > 0]
        self.pos_z_ixs = list(fg_coords[2]) if self.dim == 3 else None  # [z for z in range(self.mask_3d.shape[2]) if self.mask_3d[..., z].max() > 0]

         # 3x4, row: xyz; col: coord_min, coord_max, coord_length, coord_center
        obj_loc_property = np.array(
                                    [self.range_in_list(list(fg_coords[i]), self.mask_3d.shape[i])
                                    # self.range_in_list(self.pos_y_ixs, self.mask_3d.shape[1]), 
                                    # self.range_in_list(self.pos_z_ixs, self.mask_3d.shape[2]) 
                                    for i in range(self.dim)
                                    ], dtype= np.int32)
        return obj_loc_property

    def range_in_list(self, ixs, slen):
        # range_in_list = lambda l: max(l) - min(l)
        ix_max = max(ixs) if len(ixs) > 0 else slen
        ix_min = min(ixs) if len(ixs) > 0 else 0
        try: 
            r = ix_max - ix_min
            c = (ix_max + ix_min)//2
        except ValueError:  
            r = slen
            c = slen//2
        return ix_min, ix_max, r, c

    def pos_idx_dim(self):

        if self.image_center:
            self.obj_shape_xyz = list(self.mask_3d.shape)
            self.obj_center_xyz = [s//2 for s in self.mask_3d.shape]
        else:
            self.obj_shape_xyz = self.obj_loc_property[:, 2]
            self.obj_center_xyz = self.obj_loc_property[:, 3]

    def extend_bbox(self,  extend = 0):
        new_min = np.array([max(a, 0) for a in (self.obj_loc_property[:, 0] - extend)])
        new_max = np.array([min(a, self.mask_3d.shape[i]) for i, a in enumerate(self.obj_loc_property[:, 1] + extend)])
        new_bbox = np.stack([new_min, new_max], axis = 1)
        return new_bbox

def clock(func):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args,  **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        # arg_str = ', '.join(repr(arg) for arg in args)
        print('%s takes %0.4fs' % (name, elapsed))
        return result
    return clocked


def create_grid_torch(
    spatial_size,
    spacing = None,
    homogeneous: bool = True,
    dtype = float,
):
    """
    compute a `spatial_size` mesh.

    Args:
        spatial_size: spatial size of the grid.
        spacing: same len as ``spatial_size``, defaults to 1.0 (dense grid).
        homogeneous: whether to make homogeneous coordinates.
        dtype: output grid data type.
    """
    spacing = spacing or tuple(1.0 for _ in spatial_size)
    ranges = [torch.linspace(-(d - 1.0) / 2.0 * s, (d - 1.0) / 2.0 * s, int(d)) for d, s in zip(spatial_size, spacing)]
    coords = torch.meshgrid(*ranges)
    if not homogeneous:
        return coords
    return torch.stack( coords + (torch.ones_like(coords[0]), ))