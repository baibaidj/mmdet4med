
import torch
from torch import nn
from mmdet.core.post_processing.bbox_nms import batched_nms_3d
from mmdet.core.bbox.clip_nn import clip_boxes_to_image
from mmdet.core.bbox.ops_nn import remove_small_boxes
from monai.utils import fall_back_tuple
from monai.data.utils import compute_importance_map
from ...utils.resize import resize_3d
import copy


class BboxSegEnsembler1Case(nn.Module):

    def __init__(self, image_size, patch_size, num_classes = 2, test_cfg = {}, device = 'cpu') -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.reset_seg_output()
        self.reset_seg_countmap()
        self.initial_tile_weight_map()
        self.det_storage = []
        self.test_cfg = test_cfg
        self.device = device


        super().__init__()

    def initial_tile_weight_map(self, *args, **kwargs):
        self.tile_weight_map = compute_importance_map(*args, device = self.device, 
                                                    **kwargs)

    def reset_seg_output(self):
        if hasattr(self, 'seg_output'):
            self.seg_output = 0
        else:
            self.seg_output = torch.zeros((1, self.num_classes) + self.image_size, 
                                            dtype=torch.half, device = self.device)

    def reset_seg_countmap(self):
        if hasattr(self, 'reset_seg_countmap'):
            self.seg_countmap = 0
        else:
            self.seg_countmap = torch.zeros((1, self.num_classes) + self.image_size, 
                                        dtype=torch.short, device = self.device)

    def _update_seg_output_1tile(self, seg_logit_tile, slice_tile):
        # self.seg_output = self.seg_output.to(seg_logit_tile.device)

        self.seg_output[slice_tile] = seg_logit_tile * self.tile_weight_map
        self.seg_countmap[slice_tile] += self.tile_weight_map
        
    def update_seg_output_batch(self, seg_logit_tiles, slice_tiles):
        for seg, sli in zip(seg_logit_tiles, slice_tiles):
            self._update_seg_output_1tile(seg, sli)

    def finalize_segmap(self, final_slicing = None):
        # final_slicing : list[slice() ] len = 5
        result = self.seg_output/self.seg_countmap
        if final_slicing is not None:
            return result[final_slicing]
        else: return result


    def store_det_output(self, det_result_tiles):
        # det_result_tiles: list[(nx7, n), (), ...]
        self.det_storage.extend(det_result_tiles)
    
    def finalize_det_bbox(self, final_slicing_spatial = None):
        """
        # final_slicing_spatial : list[slice() ] len = 3
        return:
            dets_clean: tuple[Tensor(bboxes, nx7), Tensor(labels n)], 
        """
        bboxes_raw = torch.cat([a[0] for a in self.det_storage], axis = 0)
        labels_raw = torch.cat([a[1] for a in self.det_storage], axis = 0)

        bboxes, labels = postprocess_detect_results((bboxes_raw, labels_raw), 
                                                    self.image_size, self.test_cfg)

        if final_slicing_spatial is not None:
            for dim, slice_dim in enumerate(final_slicing_spatial):
                bboxes[:, [dim, dim + 3]] += - slice_dim[0]
                bboxes[:, [dim, dim + 3]] = torch.max(bboxes[:, [dim, dim + 3]], slice_dim[1])
        return (bboxes, labels)


    def offset_preprocess4detect(self, det_result, img_meta, ready_img_shape, rescale):
        """
        store inplace
        preprocess pipeline: resize, padding, flip
        """

        this_bboxes = det_result[0] # (bboxes_nx7, labels_nx1)

        img1info = img_meta['img_meta_dict']
        origin_shape = img1info['spatial_shape']
        shape_post_resize = img1info.get('shape_post_resize', origin_shape)
        shape_post_pad = img1info.get('padshape', shape_post_resize) # pad to end

        # 1. accounting for flip
        flip = img1info.get('flip', False)
        flip_direction = img1info.get('flip_direction', None)
        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None

        if flip and (flip_dims is not None): 
            for fdim in flip_dims:
                this_bboxes[:, [fdim, fdim + 3]] = ready_img_shape[fdim] - this_bboxes[:, [fdim, fdim + 3]]

        # 2, accounting for padding
        # print_tensor(f'\t[SlideInfer] ShapePostResize:{shape_post_resize} OriginShape: {origin_shape} Preds:', preds)       
        this_bboxes = clip_boxes_to_image(this_bboxes, shape_post_pad)

        # 3. accounting for resizing
        # print_tensor('count for pad', preds)
        if img1info.get('new_pixdim', False):
            if rescale and (ready_img_shape != origin_shape):
                for i, (f, o) in  enumerate(zip(shape_post_resize, origin_shape)):
                    this_bboxes[:, [i, i+3]] = this_bboxes[:, [i, i+3]] * o/f
                this_bboxes = clip_boxes_to_image(this_bboxes, origin_shape)

        det_result[0] = this_bboxes
        return det_result

    def offset_preprocess4seg(self, seg_result, img_meta, ready_img_shape, rescale):
        """
        store inplace
        preprocess pipeline: origin_shape > resize, padding, flip > ready_img_shape
        """
        this_seg_5d = seg_result

        img1info = img_meta['img_meta_dict']
        origin_shape = img1info['spatial_shape']
        shape_post_resize = img1info.get('shape_post_resize', origin_shape)
        shape_post_pad = img1info.get('padshape', shape_post_resize) # pad to end

        # 1. accounting for flip
        flip = img1info.get('flip', False)
        flip_direction = img1info.get('flip_direction', None)
        if flip_direction == 'diagonal': flip_dims = (0, 1)
        elif flip_direction == 'headfeet': flip_dims = (2, )
        elif flip_direction == 'all3axis': flip_dims = (0, 1, 2)
        else: flip_dims = None

        if flip and (flip_dims is not None): 
            this_seg_5d = this_seg_5d.flip(dims = [a + 2 for a in flip_dims])

        # 2, accounting for padding
        # print_tensor(f'\t[SlideInfer] ShapePostResize:{shape_post_resize} OriginShape: {origin_shape} Preds:', preds)       
        this_seg_5d = this_seg_5d[...,:shape_post_pad[-3], 
                                        :shape_post_pad[-2], 
                                        :shape_post_pad[-1]]

        # 3. accounting for resizing
        if img1info.get('new_pixdim', False):
            if rescale and (shape_post_resize != origin_shape):
                this_seg_5d = resize_3d(this_seg_5d, size=origin_shape,
                                        mode=self.get_output_mode(), #'bilinear'
                                        align_corners=self.align_corners,
                                        warning=False)
        return this_seg_5d


def postprocess_detect_results(det_tuple, img_shape, test_cfg):
    """
    Args:
        det_tuples: list[tuple[nx7, n]], bboxes, labels
        cfg.score_thr, cfg.nms, cfg.max_per_img
    
    """
    test_cfg = copy(test_cfg)
    max_per_img = test_cfg.pop('max_per_img', 100)
    # 1. top 
    from mmdet.core.export import get_k_for_topk
    bbox_nx7, label_n = det_tuple
    topk = test_cfg.pop('nms_pre_tiles', 1000)
    topk = get_k_for_topk(topk, det_tuple.shape[0])

    _, topk_idx = bbox_nx7[:, -1].topk(topk)
    bbox_nx7, label_n = bbox_nx7[topk_idx, ...], label_n[topk_idx]

    # 2. score threshold
    # NonZero not supported  in TensorRT
    valid_mask = bbox_nx7[:, 6] >  test_cfg.pop('score_thr', 0.01) # score_thr
    bbox_nx7, label_n = bbox_nx7[valid_mask], label_n[valid_mask]

    # 3. clipping
    bbox_nx7[:, :6] = clip_boxes_to_image(bbox_nx7[:, :6], img_shape, is_xyz = True)

    # 4. remove small boxes
    keep = remove_small_boxes(bbox_nx7[:, :6], min_size=test_cfg.pop('min_bbox_size', 0.01) )
    bbox_nx7, label_n = bbox_nx7[keep], label_n[keep]

    # 5. nms
    dets_clean, keep_idx = batched_nms_3d(bbox_nx7[:, :6], bbox_nx7[:, 6], 
                                         label_n, test_cfg['nms'])
    dets_clean = torch.cat([dets_clean, bbox_nx7[keep_idx, -1]], axis = 1)
    label_n = label_n[keep_idx]

    if max_per_img > 0:
        dets_clean = dets_clean[:max_per_img]
        label_n = label_n[:max_per_img]

    return dets_clean, label_n

class ShapeHolder:

    def __init__(self, ready_img_shape, patch_shape) -> None:
        self.ready_img_shape = ready_img_shape # (HWD)
        self.spatial_dim = len(ready_img_shape)

        self.patch_shape = fall_back_tuple(patch_shape, ready_img_shape)

        # # in case that image size is smaller than roi size
        self.ready_shape_safe = tuple(max(self.ready_img_shape[i], self.patch_shape[i]) 
                                            for i in range(self.spatial_dim))

        # torch padding from last dim to first dim, backward
        self.pad_shape = []
        for k in reversed(range(self.spatial_dim)):
            diff = max(self.patch_shape[k] - self.ready_img_shape[k], 0)
            half = diff // 2
            self.pad_shape.extend([half, diff - half])
        
        self._get_final_slicing()

    def _get_final_slicing(self):
        
        self.final_slicing = []
        for sp in range(self.spatial_dim): # 0 2 4 
            slice_dim = slice(self.pad_shape[sp * 2], 
                              self.ready_img_shape[self.spatial_dim - sp - 1] + self.pad_shape[sp * 2])
            self.final_slicing.insert(0, slice_dim)
        self.final_slicing_spatial = self.final_slicing
        while len(self.final_slicing) < len(self.ready_shape_safe.shape):
            self.final_slicing.insert(0, slice(None))
        
