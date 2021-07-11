"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""
import numpy as np
import logging
from monai.networks.blocks.nms_ import compute_overlaps


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes



def generate_anchors_3D(scales_xy, scales_z, ratios, shape, feature_stride_xy, feature_stride_z, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios

    scales_xy, ratios_meshed = np.meshgrid(np.array(scales_xy), np.array(ratios))
    scales_xy = scales_xy.flatten()
    ratios_meshed = ratios_meshed.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales_xy / np.sqrt(ratios_meshed)
    widths = scales_xy * np.sqrt(ratios_meshed)
    depths = np.tile(np.array(scales_z), len(ratios_meshed)//np.array(scales_z)[..., None].shape[0])

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride_xy #translate from fm positions to input coords.
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride_xy
    shifts_z = np.arange(0, shape[2], anchor_stride) * (feature_stride_z)
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (y, x, z) and a list of (h, w, d)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack([box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, y2, x2, z1, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    boxes = np.transpose(np.array([boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 2], boxes[:, 5]]), axes=(1, 0))
    return boxes


def generate_pyramid_anchors(
        dim,
        patch_size,
        pyramid_levels=None,
        ratios=[0.5, 1, 2],
        anchor_stride=1,
):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    :param scales: RPN_ANCHOR_SCALES , e.g. [4, 8, 16, 32]
    :param ratios: RPN_ANCHOR_RATIOS , e.g. [0.5, 1, 2]
    :param feature_shapes: BACKBONE_SHAPES , e.g.  [array of shapes per feature map] [80, 40, 20, 10, 5]
    :param feature_strides: BACKBONE_STRIDES , e.g. [2, 4, 8, 16, 32, 64]
    :param anchors_stride: RPN_ANCHOR_STRIDE , e.g. 1
    :return anchors: (N, (y1, x1, y2, x2, (z1), (z2)). All generated anchors in one array. Sorted
    with the same order of the given scales. So, anchors of scale[0] come first, then anchors of scale[1], and so on.
    """
    if pyramid_levels is None:
        pyramid_levels = [0, 1, 2, 3]
    # backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}  # original_param
    backbone_strides = {'xy': [4, 8, 16, 32], 'z': [4, 8, 16, 32]}
    scales = {'xy': [[8], [16], [32], [64]], 'z': [[8], [16], [32], [64]]}
    scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in scales['xy']]
    scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in scales['z']]

    if dim == 2:
        backbone_shapes = np.array(
            [[int(np.ceil(patch_size[0] / stride)),
              int(np.ceil(patch_size[1] / stride))]
             for stride in backbone_strides['xy']])
    else:
        backbone_shapes = np.array(
            [[int(np.ceil(patch_size[0] / stride)),
              int(np.ceil(patch_size[1] / stride)),
              int(np.ceil(patch_size[2] / stride_z))]
             for stride, stride_z in zip(backbone_strides['xy'], backbone_strides['z'])])

    feature_shapes = backbone_shapes
    feature_strides = backbone_strides

    anchors = []
    logging.info("feature map shapes: {}".format(feature_shapes))
    logging.info("anchor scales: {}".format(scales))

    expected_anchors = [np.prod(feature_shapes[ii]) * len(ratios) * len(scales['xy'][ii]) for ii in pyramid_levels]

    for lix, level in enumerate(pyramid_levels):
        if len(feature_shapes[level]) == 2:
            anchors.append(generate_anchors(scales['xy'][level], ratios, feature_shapes[level],
                                            feature_strides['xy'][level], anchor_stride))
        else:
            anchors.append(generate_anchors_3D(scales['xy'][level], scales['z'][level], ratios, feature_shapes[level],
                                            feature_strides['xy'][level], feature_strides['z'][level], anchor_stride))

        logging.info("level {}: built anchors {} / expected anchors {} ||| total build {} / total expected {}".format(
            level, anchors[-1].shape, expected_anchors[lix], np.concatenate(anchors).shape, np.sum(expected_anchors)))

    out_anchors = np.concatenate(anchors, axis=0)
    return out_anchors


def gt_anchor_matching(dim, anchors, gt_boxes, rpn_bbox_std_dev, anchor_matching_iou=0.7, rpn_train_anchors_per_image=6, gt_class_ids=None):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2, (z1), (z2))]
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2, (z1), (z2))]
    gt_class_ids (optional): [num_gt_boxes] Integer class IDs for one stage detectors. in RPN case of Mask R-CNN,
    set all positive matches to 1 (foreground)

    Returns:
    anchor_class_matches: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral.
               In case of one stage detectors like RetinaNet/RetinaUNet this flag takes
               class_ids as positive anchor values, i.e. values >= 1!
    anchor_delta_targets: [N, (dy, dx, (dz), log(dh), log(dw), (log(dd)))] Anchor bbox deltas.
    """

    anchor_class_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    anchor_delta_targets = np.zeros((rpn_train_anchors_per_image, 2*dim))

    if gt_boxes is None:
        anchor_class_matches = np.full(anchor_class_matches.shape, fill_value=-1)
        return anchor_class_matches, anchor_delta_targets

    # for mrcnn: anchor matching is done for RPN loss, so positive labels are all 1 (foreground)
    if gt_class_ids is None:
        gt_class_ids = np.array([1] * len(gt_boxes))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= anchor_matching_iou then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.1 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.1).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    if anchors.shape[1] == 4:
        anchor_class_matches[(anchor_iou_max < 0.1)] = -1
    elif anchors.shape[1] == 6:
        anchor_class_matches[(anchor_iou_max < 0.01)] = -1
    else:
        raise ValueError('anchor shape wrong {}'.format(anchors.shape))

    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    for ix, ii in enumerate(gt_iou_argmax):
        anchor_class_matches[ii] = gt_class_ids[ix]

    # 3. Set anchors with high overlap as positive.
    above_trhesh_ixs = np.argwhere(anchor_iou_max >= anchor_matching_iou)
    anchor_class_matches[above_trhesh_ixs] = gt_class_ids[anchor_iou_argmax[above_trhesh_ixs]]

    # Subsample to balance positive anchors.
    ids = np.where(anchor_class_matches > 0)[0]
    extra = len(ids) - (rpn_train_anchors_per_image // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        anchor_class_matches[ids] = 0

    # Leave all negative proposals negative now and sample from them in online hard example mining.
    # For positive anchors, compute shift and scale needed to transform them to match the corresponding GT boxes.
    ids = np.where(anchor_class_matches > 0)[0]
    ix = 0  # index into anchor_delta_targets

    for i, a in zip(ids, anchors[ids]):
        # closest gt box (it might have IoU < anchor_matching_iou)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # convert coordinates to center plus width/height.
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        if dim == 2:
            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]

        else:
            gt_d = gt[5] - gt[4]
            gt_center_z = gt[4] + 0.5 * gt_d
            a_d = a[5] - a[4]
            a_center_z = a[4] + 0.5 * a_d

            anchor_delta_targets[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                (gt_center_z - a_center_z) / a_d,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
                np.log(gt_d / a_d)
            ]

        # normalize.
        anchor_delta_targets[ix] /= rpn_bbox_std_dev
        ix += 1

    return anchor_class_matches, anchor_delta_targets