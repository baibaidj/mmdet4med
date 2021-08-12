import mmcv
import numpy as np
import torch, pdb

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWHBBoxCoder3D(BaseBBoxCoder):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, z1, x2, y2, z2) into delta (dx, dy, yz, dw, dh, dd) and
    decodes delta (dx, dy, dz, dw, dh, dd) back to original bbox (x1, y1, z1, x2, y2, z2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1.),
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        assert len(target_means) == 6 and len(target_stds) == 6
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 6
        encoded_bboxes = bbox2delta3d(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 6) or (N, 6)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 6) or (B, N, 6) or
               (N, num_classes * 6) or (N, 6). Note N = num_anchors * W * H * D
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, D, C) or (H, W, D). If bboxes shape is (B, N, 6), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height. #NOTE??

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1) 
        decoded_bboxes = delta2bbox3d(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, self.clip_border,
                                    self.add_ctr_clamp, self.ctr_clamp)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bbox2delta3d(proposals, gt, means=(0., 0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, z, w, h, d of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        # left upper corner and right lower corner
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 6)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 6)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 6), where columns represent 
                dx, dy, dz, dw, dh, dd.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 3]) * 0.5 # center x
    py = (proposals[..., 1] + proposals[..., 4]) * 0.5 # center y
    pz = (proposals[..., 2] + proposals[..., 5]) * 0.5 # center z
    pw = proposals[..., 3] - proposals[..., 0] 
    ph = proposals[..., 4] - proposals[..., 1]
    pd = proposals[..., 5] - proposals[..., 2]

    gx = (gt[..., 0] + gt[..., 3]) * 0.5
    gy = (gt[..., 1] + gt[..., 4]) * 0.5
    gz = (gt[..., 2] + gt[..., 5]) * 0.5
    gw = gt[..., 3] - gt[..., 0]
    gh = gt[..., 4] - gt[..., 1]
    gd = gt[..., 5] - gt[..., 2]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dz = (gz - pz) / pd

    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dd = torch.log(gd / pd)

    deltas = torch.stack([dx, dy, dz, dw, dh, dd], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


@mmcv.jit(coderize=True)
def delta2bbox3d(rois,
               deltas,
               means=(0., 0., 0., 0., 0., 0.),
               stds=(1., 1., 1., 1., 1., 1.0),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.
    #NOTE: in sliding window inference, clip border may not be necessary for all windows
    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 6) or (B, N, 6)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 6) or (B, N, 6) or
            (N, num_classes * 6) or (N, 6). Note N = num_anchors * W * H * D
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, D, C) or (H, W, D). If rois shape is (B, N, 6), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 6) or (B, N, 6) or
           (N, num_classes * 6) or (N, 6), where 6 represent
           tl_x, tl_y, tlz, br_x, br_y, br_z. # tl: top left, br: bottom right

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    # account for multi class, last dim: num_classes * 6
    means = deltas.new_tensor(means).view(1, -1).repeat(1,deltas.size(-1) // 6) 
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 6)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[..., 0::6] # every 6 element
    dy = denorm_deltas[..., 1::6]
    dz = denorm_deltas[..., 2::6]
    dw = denorm_deltas[..., 3::6]
    dh = denorm_deltas[..., 4::6]
    dd = denorm_deltas[..., 5::6]

    x1, y1, z1 = rois[..., 0], rois[..., 1], rois[..., 2]
    x2, y2, z2 = rois[..., 3], rois[..., 4], rois[..., 5]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx) # 
    py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy)
    pz = ((z1 + z2) * 0.5).unsqueeze(-1).expand_as(dz)
    # Compute width/height/depth of each roi
    pw = (x2 - x1).unsqueeze(-1).expand_as(dw)
    ph = (y2 - y1).unsqueeze(-1).expand_as(dh)
    pd = (z2 - z1).unsqueeze(-1).expand_as(dd)

    dx_width = pw * dx
    dy_height = ph * dy
    dz_depth = pd * dz

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = torch.clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = torch.clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dz_depth = torch.clamp(dz_depth, max=ctr_clamp, min=-ctr_clamp)
        dw = torch.clamp(dw, max=max_ratio)
        dh = torch.clamp(dh, max=max_ratio)
        dd = torch.clamp(dd, max=max_ratio)
    else:
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        dd = dd.clamp(min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gd = pd * dd.exp()
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    gz = pz + dz_depth
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    z1 = gz - gd * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    z2 = gz + gd * 0.5

    bboxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1).view(deltas.size())

    if clip_border and max_shape is not None:
        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, z1, x2, y2, z2 = dynamic_clip_for_onnx(x1, y1, z1, x2, y2, z2, max_shape)
            bboxes = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1).view(deltas.size())
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :3].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)
        min_xyz = x1.new_tensor(0)
        max_xyz = torch.cat([max_shape] * (deltas.size(-1) // 3), dim=-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xyz, min_xyz, bboxes)
        bboxes = torch.where(bboxes > max_xyz, max_xyz, bboxes)

    return bboxes
