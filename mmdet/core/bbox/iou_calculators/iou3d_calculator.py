import torch
from mmdet.utils import print_tensor
from .builder import IOU_CALCULATORS
import pdb

def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


@IOU_CALCULATORS.register_module()
class BboxOverlaps3D:
    """3D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 3D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 6) in <x1, y1, z1, x2, y2, z2>
                format, or shape (m, 7) in <x1, y1, z1, x2, y2, z2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 6) in <x1, y1, z1, x2, y2, z2>
                format, or shape (m, 7) in <x1, y1, z1, x2, y2, z2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 6, 7]
        assert bboxes2.size(-1) in [0, 6, 7]
        if bboxes2.size(-1) == 7:
            bboxes2 = bboxes2[..., :6]
        if bboxes1.size(-1) == 7:
            bboxes1 = bboxes1[..., :6]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps_3d(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps_3d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 6, bboxes2 is N x 6, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2 # left top?
            rb: M x N x 2 # right bottom?
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte, # why 9?

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2> format or empty.
        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False. boxes have been paired (n pairs) and no need for MxN pairs. 
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 6
    assert (bboxes1.size(-1) == 6 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 6 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    length_in_box = lambda x, dim1, dim0: (x[..., dim1] - x[..., dim0])
    # area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    # area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * ( bboxes2[..., 3] - bboxes2[..., 1])
    area1 = length_in_box(bboxes1, 3, 0) * length_in_box(bboxes1, 4, 1) * length_in_box(bboxes1, 5, 2)
    area2 = length_in_box(bboxes2, 3, 0) * length_in_box(bboxes2, 4, 1) * length_in_box(bboxes2, 5, 2)
    # area1_torch = torch.prod(bboxes1[:, 3:6] - bboxes1[:, 0:3], dim = 1)
    # print_tensor('[IOU] a1', area1)
    # print_tensor('[IOU] a2', area2)

    if is_aligned:
        lt = torch.maximum(bboxes1[..., :3], bboxes2[..., :3])  # [B, rows, 3] # left top
        rb = torch.minimum(bboxes1[..., 3:], bboxes2[..., 3:])  # [B, rows, 3] # right bottom

        whd = fp16_clamp(rb - lt, min=0)
        overlap = whd[..., 0] * whd[..., 1] * whd[..., 2]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.maximum(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.maximum(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])  # [B, rows, cols, 3]
        rb = torch.minimum(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])  # [B, rows, cols, 3]

        whd = fp16_clamp(rb - lt, min=0) 
        overlap = whd[..., 0] * whd[..., 1] * whd[..., 2]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.minimum(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
            enclosed_rb = torch.maximum(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] *  enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
