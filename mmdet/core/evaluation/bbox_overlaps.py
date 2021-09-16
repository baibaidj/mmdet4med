# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pdb


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 6)
        bboxes2(ndarray): shape (k, 6)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """
    bbox_volume_nx6 = lambda x: (x[:, 5] - x[:, 2]) * (x[:, 4] - x[:, 1]) * (x[:, 3] - x[:, 0])
    assert mode in ['iou', 'iof']
    length_in_box = lambda x, dim1, dim0: (x[..., dim1] - x[..., dim0])

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = length_in_box(bboxes1, 3, 0) * length_in_box(bboxes1, 4, 1) * length_in_box(bboxes1, 5, 2)
    area2 = length_in_box(bboxes2, 3, 0) * length_in_box(bboxes2, 4, 1) * length_in_box(bboxes2, 5, 2)
    # inner_start_dim = lambda b1, b2, dim: np.maximum(b1[dim], b2[dim])
    # inner_end_dim = lambda b1, b2, dim: np.minimum(b1[dim], b2[dim])
    lt = np.maximum(bboxes1[:, None, :3], bboxes2[None, :, :3])  # [B, rows, cols, 3]
    rb = np.minimum(bboxes1[:, None, 3:], bboxes2[None, :, 3:])  # [B, rows, cols, 3]
    whd = np.clip(rb - lt, 0, 1024) # nxkx3
    overlap = whd[..., 0] * whd[..., 1] * whd[..., 2] # nxk
    if mode == 'iou':
        union = area1[..., None] + area2[..., None, :] - overlap # nx1, 1xk, nxk, 
    else:
        union = area1[..., None] if not exchange else area2[..., None]
    union = np.maximum(union, eps)
    ious = overlap / union
    # for i in range(bboxes1.shape[0]):
    #     x_start = inner_start_dim(bboxes1[i], bboxes2[i], 0)#np.maximum(bboxes1[i, 0], bboxes2[:, 0])
    #     y_start = inner_start_dim(bboxes1[i], bboxes2[i], 1) #np.maximum(bboxes1[i, 1], bboxes2[:, 1])
    #     z_start = inner_start_dim(bboxes1[i], bboxes2[i], 2)

    #     x_end = inner_end_dim(bboxes1[i], bboxes2[i], 3) # np.minimum(bboxes1[i, 2], bboxes2[:, 2])
    #     y_end = inner_end_dim(bboxes1[i], bboxes2[i], 4) #np.minimum(bboxes1[i, 3], bboxes2[:, 3])
    #     z_end = inner_end_dim(bboxes1[i], bboxes2[i], 5)
    #     pdb.set_trace()
    #     overlap = np.maximum(x_end - x_start, 0) * \
    #                     np.maximum(y_end - y_start, 0) * \
    #                             np.maximum(z_end - z_start, 0)
    #     if mode == 'iou':
    #         union = area1[i] + area2 - overlap
    #     else:
    #         union = area1[i] if not exchange else area2
    #     union = np.maximum(union, eps)
    #     ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
