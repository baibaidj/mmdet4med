"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

import numpy as np
import scipy.misc
import scipy.ndimage
import torch
from torch.autograd import Variable
import torch.nn as nn


def nms_2D(dets, thresh):
    if torch.is_tensor(dets):
        dets = dets.detach().cpu().numpy()

    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    # order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_3D(dets, thresh):
    if torch.is_tensor(dets):
        dets = dets.detach().cpu().numpy()

    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    z1 = dets[:, 4]
    z2 = dets[:, 5]
    scores = dets[:, 6]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        zz1 = np.maximum(z1[i], z1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        zz2 = np.minimum(z2[i], z2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        d = np.maximum(0.0, zz2 - zz1 + 1)
        inter = w * h * d
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep



############################################################
#  Bounding Boxes
############################################################


def compute_iou_2D(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2] THIS IS THE GT BOX
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union

    return iou



def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2, z1, z2] (typically gt box)
    boxes: [boxes_count, (y1, x1, y2, x2, z1, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou



def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. / 3D: (z1, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    if boxes1.shape[1] == 4:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i] #this is the gt box
            overlaps[:, i] = compute_iou_2D(box2, boxes1, area2[i], area1)
        return overlaps

    else:
        # Areas of anchors and GT boxes
        volume1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 4])
        volume2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 4])
        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]  # this is the gt box
            overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
        return overlaps



def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)] / 3D: (z1, z2))
    """
    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)
    result = torch.stack([dy, dx, dh, dw], dim=1)

    if box.shape[1] > 4:
        depth = box[:, 5] - box[:, 4]
        center_z = box[:, 4] + 0.5 * depth
        gt_depth = gt_box[:, 5] - gt_box[:, 4]
        gt_center_z = gt_box[:, 4] + 0.5 * gt_depth
        dz = (gt_center_z - center_z) / depth
        dd = torch.log(gt_depth / depth)
        result = torch.stack([dy, dx, dz, dh, dw, dd], dim=1)

    return result



def unmold_mask_2D(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox
    out_zoom = [y2 - y1, x2 - x1]
    zoom_factor = [i / j for i, j in zip(out_zoom, mask.shape)]
    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2])
    full_mask[y1:y2, x1:x2] = mask
    return full_mask



def unmold_mask_3D(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2, z1, z2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    y1, x1, y2, x2, z1, z2 = bbox
    out_zoom = [y2 - y1, x2 - x1, z2 - z1]
    zoom_factor = [i/j for i,j in zip(out_zoom, mask.shape)]
    mask = scipy.ndimage.zoom(mask, zoom_factor, order=1).astype(np.float32)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:3])
    full_mask[y1:y2, x1:x2, z1:z2] = mask
    return full_mask


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


def apply_box_deltas_2D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result



def apply_box_deltas_3D(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is y1, x1, y2, x2, z1, z2
    deltas: [N, 6] where each row is [dy, dx, dz, log(dh), log(dw), log(dd)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 4]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 4] + 0.5 * depth
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    height *= torch.exp(deltas[:, 3])
    width *= torch.exp(deltas[:, 4])
    depth *= torch.exp(deltas[:, 5])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    result = torch.stack([y1, x1, y2, x2, z1, z2], dim=1)
    return result



def clip_boxes_2D(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def clip_boxes_3D(boxes, window):
    """
    boxes: [N, 6] each col is y1, x1, y2, x2, z1, z2
    window: [6] in the form y1, x1, y2, x2, z1, z2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3])),
         boxes[:, 4].clamp(float(window[4]), float(window[5])),
         boxes[:, 5].clamp(float(window[4]), float(window[5]))], 1)
    return boxes



def clip_boxes_numpy(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2 / [N, 6] in 3D.
    window: iamge shape (y, x, (z))
    """
    if boxes.shape[1] == 4:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
            np.clip(boxes[:, 1], 0, window[0])[:, None],
            np.clip(boxes[:, 2], 0, window[1])[:, None],
            np.clip(boxes[:, 3], 0, window[1])[:, None]), 1
        )

    else:
        boxes = np.concatenate(
            (np.clip(boxes[:, 0], 0, window[0])[:, None],
             np.clip(boxes[:, 1], 0, window[0])[:, None],
             np.clip(boxes[:, 2], 0, window[1])[:, None],
             np.clip(boxes[:, 3], 0, window[1])[:, None],
             np.clip(boxes[:, 4], 0, window[2])[:, None],
             np.clip(boxes[:, 5], 0, window[2])[:, None]), 1
        )

    return boxes



def bbox_overlaps_2D(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps



def bbox_overlaps_3D(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2, z1, z2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,6)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2, b1_z1, b1_z2 = boxes1.chunk(6, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2, b2_z1, b2_z2 = boxes2.chunk(6, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    z1 = torch.max(b1_z1, b2_z1)[:, 0]
    z2 = torch.min(b1_z2, b2_z2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros) * torch.max(z2 - z1, zeros)

    # 3. Compute unions
    b1_volume = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)  * (b1_z2 - b1_z1)
    b2_volume = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)  * (b2_z2 - b2_z1)
    union = b1_volume[:,0] + b2_volume[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)
    return overlaps


def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2) / 3D: (z1, z2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]  / 3D: (z1, z2)
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    if boxes.shape[1] > 5:
        boxes[:, 4] = boxes[:, 4].clamp(float(window[4]), float(window[5]))
        boxes[:, 5] = boxes[:, 5].clamp(float(window[4]), float(window[5]))

    return boxes


############################################################
#  Pytorch Utility Functions
############################################################


def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    unique_bool = unique_bool.type(torch.bool)
    return tensor[unique_bool.data]



def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2



def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort(descending=True)[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]



def shem(roi_probs_neg, negative_count, ohem_poolsize):
    """
    stochastic hard example mining: from a list of indices (referring to non-matched predictions),
    determine a pool of highest scoring (worst false positives) of size negative_count*ohem_poolsize.
    Then, sample n (= negative_count) predictions of this pool as negative examples for loss.
    :param roi_probs_neg: tensor of shape (n_predictions, n_classes).
    :param negative_count: int.
    :param ohem_poolsize: int.
    :return: (negative_count).  indices refer to the positions in roi_probs_neg. If pool smaller than expected due to
    limited negative proposals availabel, this function will return sampled indices of number < negative_count without
    throwing an error.
    """
    # sort according to higehst foreground score.
    probs, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = torch.tensor((ohem_poolsize * int(negative_count), order.size()[0])).min().int()
    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0])
    if torch.cuda.is_available():
        out = pool_indices[rand_idx[:negative_count].cuda()]
    else:
        out = pool_indices[rand_idx[:negative_count]]
    return out



# def initialize_weights(net):
#     """
#    Initialize model weights. Current Default in Pytorch (version 0.4.1) is initialization from a uniform distriubtion.
#    Will expectably be changed to kaiming_uniform in future versions.
#    """
#     init_type = net.cf.weight_init
#
#     for m in [module for module in net.modules() if type(module) in [nn.Conv2d, nn.Conv3d,
#                                                                      nn.ConvTranspose2d,
#                                                                      nn.ConvTranspose3d,
#                                                                      nn.Linear]]:
#         if init_type == 'xavier_uniform':
#             nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#         elif init_type == 'xavier_normal':
#             nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#         elif init_type == "kaiming_uniform":
#             nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
#             if m.bias is not None:
#                 fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
#                 bound = 1 / np.sqrt(fan_out)
#                 nn.init.uniform_(m.bias, -bound, bound)
#
#         elif init_type == "kaiming_normal":
#             nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
#             if m.bias is not None:
#                 fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
#                 bound = 1 / np.sqrt(fan_out)
#                 nn.init.normal_(m.bias, -bound, bound)



class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)

        return conv



def get_one_hot_encoding(y, n_classes):
    """
    transform a numpy label array to a one-hot array of the same shape.
    :param y: array of shape (b, 1, y, x, (z)).
    :param n_classes: int, number of classes to unfold in one-hot encoding.
    :return y_ohe: array of shape (b, n_classes, y, x, (z))
    """
    dim = len(y.shape) - 2
    if dim == 2:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3])).astype('int32')
    if dim ==3:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3], y.shape[4])).astype('int32')
    for cl in range(n_classes):
        y_ohe[:, cl][y[:, 0] == cl] = 1
    return y_ohe



def get_dice_per_batch_and_class(pred, y, n_classes):
    '''
    computes dice scores per batch instance and class.
    :param pred: prediction array of shape (b, 1, y, x, (z)) (e.g. softmax prediction with argmax over dim 1)
    :param y: ground truth array of shape (b, 1, y, x, (z)) (contains int [0, ..., n_classes]
    :param n_classes: int
    :return: dice scores of shape (b, c)
    '''
    pred = get_one_hot_encoding(pred, n_classes)
    y = get_one_hot_encoding(y, n_classes)
    axes = tuple(range(2, len(pred.shape)))
    intersect = np.sum(pred*y, axis=axes)
    denominator = np.sum(pred, axis=axes)+np.sum(y, axis=axes) + 1e-8
    dice = 2.0*intersect / denominator
    return dice



def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input



def batch_dice(pred, y, false_positive_weight=1.0, smooth=1e-6):
    '''
    compute soft dice over batch. this is a differentiable score and can be used as a loss function.
    only dice scores of foreground classes are returned, since training typically
    does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
    This way, single patches with missing foreground classes can not produce faulty gradients.
    :param pred: (b, c, y, x, (z)), softmax probabilities (network output). (c==classes)
    :param y: (b, c, y, x, (z)), one-hot-encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float). This function discards the background score and returns the mean of foreground scores.
    '''
    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2 * intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    else:
        raise ValueError('wrong input dimension in dice loss')




def batch_dice_mask(pred, y, mask, false_positive_weight=1.0, smooth=1e-6):
    '''
    compute soft dice over batch. this is a diffrentiable score and can be used as a loss function.
    only dice scores of foreground classes are returned, since training typically
    does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
    This way, single patches with missing foreground classes can not produce faulty gradients.
    :param pred: (b, c, y, x, (z)), softmax probabilities (network output).
    :param y: (b, c, y, x, (z)), one hote encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float). This function discards the background score and returns the mean of foreground scores.
    '''

    mask = mask.unsqueeze(1).repeat(1, 2, 1, 1)

    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y * mask, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred * mask + y * mask, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth))[1:]) # only fg dice here.

    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    else:
        raise ValueError('wrong input dimension in dice loss')