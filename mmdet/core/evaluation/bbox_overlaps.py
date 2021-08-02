import numpy as np


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

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
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
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
    area1 = bbox_volume_nx6(bboxes1)
    area2 = bbox_volume_nx6(bboxes2)

    inner_start_dim = lambda b1, b2, dim: np.maximum(b1[dim], b2[dim])
    inner_end_dim = lambda b1, b2, dim: np.minimum(b1[dim], b2[dim])
    for i in range(bboxes1.shape[0]):
        x_start = inner_start_dim(bboxes1[i], bboxes2[i], 0)#np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = inner_start_dim(bboxes1[i], bboxes2[i], 1) #np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        z_start = inner_start_dim(bboxes1[i], bboxes2[i], 2)

        x_end = inner_end_dim(bboxes1[i], bboxes2[i], 3) # np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = inner_end_dim(bboxes1[i], bboxes2[i], 4) #np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        z_end = inner_end_dim(bboxes1[i], bboxes2[i], 5)

        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0) * np.maximum(z_end - z_start)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
