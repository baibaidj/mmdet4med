"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""
from typing import Any, Callable, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from monai.data.utils import compute_importance_map, get_valid_patch_size, dense_patch_slices
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple
import math
from monai.utils import first, ensure_tuple_size
from functools import reduce
import operator
from monai.networks.blocks.nms_ import nms_2D, nms_3D
from monai.networks.nets.retina_net_ import refine_det

__all__ = ["sliding_window_multi_pyramid_inference", "sliding_window_detection"]


def dense_multi_pyramid_patch_slices(
        image_size: Sequence[int],
        patch_size: Sequence[int],
        scan_interval: Sequence[int],
        pyramid_scale: Union[Sequence[float], float],
) -> List[List[Tuple[slice, ...]]]:
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval
        pyramid_scale: pyramid_scale for multiple pyramid input, includes scale factor of origin patch size, e.g. [1.0, 1.5, 2.0]
        offset: offset for each point position
    Returns:
        a list of slice objects defining each patch

    """
    num_spatial_dims = len(image_size)
    patch_size = get_valid_patch_size(image_size, patch_size)
    scan_interval = ensure_tuple_size(scan_interval, num_spatial_dims)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    if isinstance(pyramid_scale, float):
        pyramid_scale = [pyramid_scale]

    pyramid_levels = len(pyramid_scale)
    outs = []
    for p in range(pyramid_levels):
        p_inv = pyramid_levels - 1 - p
        patch_size_ = np.asarray(patch_size) * pyramid_scale[p]
        patch_size_ = patch_size_.astype(np.int)
        image_size_ = np.asarray(image_size) + (pyramid_scale[p] - 1.0) * np.asarray(patch_size)
        image_size_ = np.ceil(image_size_).astype(np.int)
        offset_ = (pyramid_scale[p_inv] - 1.0) * np.asarray(patch_size) * 0.5
        offset_ = np.round(offset_).astype(np.int)
        starts = []
        for dim in range(num_spatial_dims):
            dim_starts = []
            for idx in range(scan_num[dim]):
                start_idx = idx * scan_interval[dim]
                start_idx -= max(start_idx + patch_size_[dim] - image_size_[dim], 0)
                dim_starts.append(start_idx + offset_[dim])  # int(offset[dim]/2 + 0.5))
            starts.append(dim_starts)
        out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
        out = [tuple(slice(s, s + patch_size_[d]) for d, s in enumerate(x)) for x in out]
        outs.append(out)
    return outs


def sliding_window_multi_pyramid_inference(
        inputs: torch.Tensor,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        pyramid_scale: Union[Sequence[float], float] = (1.0,),
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on multiple pyramid `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        pyramid_scale: pyramid_scale for multiple pyramid input, includes scale factor of origin patch size, e.g. [1.0, 1.5, 2.0]
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if isinstance(pyramid_scale, float):
        pyramid_scale = [pyramid_scale]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    max_pyramid_size = max(pyramid_scale)
    offset = np.asarray(roi_size) * max(0, (max_pyramid_size - 1.0))
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0) + int(offset[k - 2] + 0.5)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)
    image_size_pad = list(inputs.shape[2:])

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices_multi_pyramid = dense_multi_pyramid_patch_slices(image_size, roi_size, scan_interval, pyramid_scale)
    num_win = len(slices_multi_pyramid[0])  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image, count_map = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [[slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win]) for slices in slices_multi_pyramid]
            for idx in slice_range
        ]

        inputs_batch = []
        for win_slice_ in unravel_slice:
            inputs_channels = []
            for win_slice in win_slice_:
                input_ = inputs[win_slice].to(sw_device)
                input_ = F.interpolate(input_, size=roi_size, mode='trilinear', align_corners=False)
                inputs_channels.append(input_)
            inputs_ = torch.cat([c for c in inputs_channels], dim=1)
            inputs_batch.append(inputs_)
        window_data = torch.cat([b for b in inputs_batch], dim=0)
        # window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation
        seg_prob_dim = len(seg_prob.shape)
        window_data_dim = len(window_data.shape)
        assert (seg_prob_dim == window_data_dim or seg_prob_dim == window_data_dim + 1)
        if seg_prob_dim == window_data_dim + 1:
            seg_prob = seg_prob[0]  # only use first level pred. result

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size_pad)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx[0]] += importance_map * seg_prob[idx - slice_g]
            count_map[original_idx[0]] += importance_map

    # account for any overlapping sections
    count_map += 1
    output_image = output_image / count_map

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]


def _get_scan_interval(
        image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def get_det_results(dim, img_shape, detections, seg_logits, box_results_list=None, model_min_confidence=0.1, unravel_slice=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                          retina_unet and dummy array for retina_net.
    """
    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, dim * 2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:

            start_point_x = unravel_slice[ix][2].start
            start_point_y = unravel_slice[ix][3].start
            if dim == 3:
                start_point_z = unravel_slice[ix][4].start
                start_pnt = [start_point_x, start_point_y, start_point_x, start_point_y, start_point_z, start_point_z]
            else:
                start_pnt = [start_point_x, start_point_y, start_point_x, start_point_y]

            boxes = detections[ix][:, :2 * dim].astype(np.int32)
            class_ids = detections[ix][:, 2 * dim + 1].astype(np.int32)
            scores = detections[ix][:, 2 * dim + 2]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if dim == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)

            boxes[:, ] += start_pnt
            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    if score >= model_min_confidence:
                        # boxes[ix2] += start_pnt
                        box_results_list[ix].append({'box_coords': boxes[ix2],
                                                     'box_score': score,
                                                     'box_type': 'det',
                                                     'box_pred_class_id': class_ids[ix2]})

    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = None  # np.zeros(img_shape)[:, 0][:, np.newaxis]
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')
    return results_dict


def sliding_window_detection(
        anchors,
        inputs: torch.Tensor,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int,
        predictor: Callable[..., torch.Tensor],
        overlap: float = 0.25,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        dim: int = 3,
        rpn_bbox_std_dev=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
        model_min_confidence: float = 0.1,
        detection_nms_threshold: float = 1e-5,
        model_max_instances_per_batch_element: int = 2000,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        *args: Any,
        **kwargs: Any,
):
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    box_info_list = []
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        class_logits, pred_deltas, seg_logits = predictor(window_data, *args, **kwargs)  # batched patch segmentation
        detections = refine_det(class_logits, pred_deltas, dim,
                                anchors, roi_size, rpn_bbox_std_dev,
                                detection_nms_threshold, model_max_instances_per_batch_element)
        results_dict = get_det_results(dim, window_data.shape, detections, seg_logits, None, model_min_confidence, unravel_slice)
        bboxes_info_ = reduce(operator.add, results_dict['boxes'])
        box_info_list.extend(bboxes_info_)

    ix_rois = []
    for b in box_info_list:
        box_coord = b['box_coords'].flatten()
        box_coord = np.append(box_coord, b['box_score'])
        ix_rois.append(box_coord)

    ix_rois = np.asarray(ix_rois)
    ix_rois = ix_rois[np.argsort(-ix_rois[:, 6])]

    # do nms
    if dim == 2:
        class_keep = nms_2D(ix_rois, thresh=detection_nms_threshold)
    else:
        class_keep = nms_3D(ix_rois, thresh=detection_nms_threshold)

    result = ix_rois[class_keep][: model_max_instances_per_batch_element]
    return result
