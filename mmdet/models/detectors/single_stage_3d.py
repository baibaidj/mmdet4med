import warnings

import torch, pdb, copy
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base3d import BaseDetector3D
from ...datasets.pipelines import Compose
from ..utils import print_tensor
from ...utils.resize import resize_3d

from mmdet.models.semantic_heads import FCNHead3D
from mmdet.core.post_processing.bbox_nms import batched_nms_3d
from mmdet.utils.resize import list_dict2dict_list
from mmdet.core import bbox2result3d, ShapeHolder, BboxSegEnsembler1Case

from monai.data.utils import dense_patch_slices
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple
from typing import Any, Callable, List, Sequence, Tuple, Union
# from profiler import 

get_meta_dict  = lambda img_meta: img_meta[0]['img_meta_dict'] if isinstance(img_meta, (list, tuple)) else img_meta['img_meta_dict']

@DETECTORS.register_module()
class SingleStageDetector3D(BaseDetector3D): 
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 seg_head = None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None, 
                 gpu_aug_pipelines = [],
                 mask2bbox_cfg = [
                    dict(type = 'FindInstances', 
                        instance_key="seg",
                        save_key="present_instances"), 
                    dict(type = 'Instances2Boxes', 
                        instance_key="seg",
                        map_key="instances",
                        box_key="gt_bboxes",
                        class_key="gt_labels",
                        present_instances="present_instances"),
                    dict(type = 'Instances2SemanticSeg', 
                        instance_key = 'seg',
                        map_key="instances",
                        seg_key = 'seg',
                        present_instances="present_instances"), 
                    ],
                verbose = False,
                 ):
        super(SingleStageDetector3D, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # pdb.set_trace()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.seg_head: FCNHead3D = build_head(seg_head) if seg_head is not None else None
        self.seg_num_classes = getattr(self.seg_head, 'cls_out_channels', 2)
        self.gpu_pipelines = Compose(gpu_aug_pipelines + mask2bbox_cfg) \
                                    if mask2bbox_cfg is not None else None
        self.verbose = verbose

        self.roi_size = self.test_cfg.pop('roi_size', None)
        self.sw_batch_size = self.test_cfg.pop('sw_batch_size', 2)
        self.overlap = self.test_cfg.pop('overlap', 0.25)
        self.blend_mode = self.test_cfg.pop('blend_mode', 'constant')
        self.padding_mode = self.test_cfg.pop('padding_mode', 'constant')
        self.sigma_scale = self.test_cfg.pop('sigma_scale', 0.125)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if self.with_seghead: mask = self.seg_head(x)
        else: mask = None
        return outs, mask

    @torch.no_grad()
    def update_img_metas(self, imgs, img_metas, seg, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # TODO: adjust keys
        gt_keys = ['img', 'gt_bboxes', 'gt_labels', 'seg'] # 'img_metas'
        data_dict = list_dict2dict_list(img_metas, verbose=False)
        data_dict.update({'img': imgs, 'seg': seg})
        data_dict = self.gpu_pipelines(data_dict)
        # for b, m in enumerate(img_metas): m['patch_shape'] = data_dict['patch_shape']
        return [data_dict[k] for k in gt_keys] 
        
    @property
    def with_seghead(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_train(self,
                      img,
                      img_metas,
                      seg,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W, D).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, tl_z, br_x, br_y, br_z] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        img, gt_bboxes, gt_labels, gt_semantic_seg = self.update_img_metas(
                                            img, img_metas, seg)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        # print_tensor('[Detector] img', img)
        # print('[Detector] gt bbox', gt_bboxes)
        # print('[Detector] gt labels', gt_labels)
        # print('[Detector] loss', losses)
        if self.with_seghead:
            # print_tensor('semantic seg', gt_semantic_seg)
            # pdb.set_trace()
            loss_seg = self.seg_head.forward_train(x, img_metas, gt_semantic_seg, 
                                                   self.train_cfg)
            # print('Detector] seg loss', loss_seg)
            losses.update(loss_seg)
        # pdb.set_trace()
        return losses

    def simple_test_tile(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W, D).
            img_metas (list[dict]): List of image information. 
            e.g.[{'img_shape': (H, W, D), 'scale_factor': 1}]
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # dense_test_mixins.BBoxTestMixin3D.simple_test_bboxes
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        if self.with_seghead: 
            mask = self.seg_head.simple_test(feat, img_metas, rescale = rescale)
        else: mask = None
        return results_list, mask

    def sliding_window_inference(self, 
        inputs: torch.Tensor,
        img_metas, 
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        rescale = True
    ) -> torch.Tensor:
        """
        TODO: visualize this method to gain insight on how the patches are cropped
        TODO: sliding has order, the target may be split during cropping, and optimal sliding order should be determined
        Sliding window inference on `inputs` with `predictor`.

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

        if (self.overlap < 0) or (self.overlap >= 1):
            raise AssertionError("overlap must be >= 0 and < 1.")
        # ip_dtype = torch.float #inputs.dtype
        device = inputs.device; sw_device = device#'cpu'
        batch_size = inputs.shape[0]
        Shaper = ShapeHolder(inputs.shape[2:], self.roi_size)
        scan_interval = _get_scan_interval(Shaper.ready_shape_safe, Shaper.patch_shape, 
                                           Shaper.spatial_dim, self.overlap)
        window_slices = dense_patch_slices(Shaper.ready_shape_safe, Shaper.patch_shape, scan_interval)
        num_win = len(window_slices)  # number of windows per image
        # Perform predictions
        # print_tensor('original inputs', inputs) BCHWD
        ensembler = BboxSegEnsembler1Case(Shaper.ready_shape_safe, Shaper.patch_shape, 
                                    self.seg_num_classes, self.test_cfg, device = device)
        ensembler.initial_tile_weight_map(Shaper.patch_shape, mode = mode, sigma_scale = sigma_scale)
        # print_tensor('[Ensemble] initalize', ensembler.seg_output_4d)
        # print_tensor('[Ensemble] initalize', ensembler.seg_countmap_4d)
        det_result_batch, seg_result_batch = [], []
        for bix in range(batch_size):
            img_meta = img_metas[bix]
            for six in range(0, num_win, self.sw_batch_size):
                slice_range = range(six, min(six + self.sw_batch_size, num_win))
                tiles_slicer = [[slice(bix, bix + 1), slice(None)] + list(window_slices[idx]) 
                                                                    for idx in slice_range]
                img_meta_tiles = [copy.deepcopy(img_meta)  for _ in range(len(slice_range))]#[img_meta] * len(slice_range)
                for i, m in enumerate(img_meta_tiles): 
                    m['tile_origin'] = [window_slices[slice_range[i]][d].start for d in range(Shaper.spatial_dim)]
                    m['img_shape'] = None #Shaper.patch_shape
                    m['scale_factor'] = [1 for _ in range(Shaper.spatial_dim * 2)]
                window_data = torch.cat([inputs[win_slice] for win_slice in tiles_slicer]).to(device)
                # print_tensor(f'[SlideInfer] window {six} data', window_data)
                det_results, seg_results = self.simple_test_tile(window_data, img_meta_tiles)  # batched patch segmentation
                # if self.verbose and six % 30 ==0 :  #self.verbose and  six % 30 ==0 
                #     print_tensor(f'\n[SliceInfer] seg results batch{bix} win{six}', seg_results)
                #     for bi, det in enumerate(det_results):
                #         print_tensor(f'\n[SliceInfer] bix {bi} det result bbox', det[0])
                #         print_tensor(f'\n[SliceInfer] bix {bi} det result score', det[1])
                # pdb.set_trace()
                ensembler.store_det_output(det_results)
                ensembler.update_seg_output_batch(seg_results, tiles_slicer)
            
            det_result_img = ensembler.finalize_det_bbox(verbose=False) # (bboxes_nx7, labels_nx1)
            seg_result_img = ensembler.finalize_segmap() # 1CHWD
            # print_tensor(f'[SliceInfer] seg results ensemble {bix}', seg_result_img)
            det_result_img = ensembler.offset_preprocess4detect(det_result_img, img_meta, 
                                                            Shaper.ready_shape_safe, rescale)
            seg_result_img = ensembler.offset_preprocess4seg(seg_result_img, img_meta, 
                                                            Shaper.ready_shape_safe, rescale)                                               
            det_result_batch.append(det_result_img)
            seg_result_batch.append(seg_result_img)
            ensembler.reset_seg_output(); ensembler.reset_seg_countmap()
            ensembler.reset_det_storage()
            torch.cuda.empty_cache()
            # print_tensor('[Ensemble] reset', ensembler.seg_output_4d)
            # print_tensor('[Ensemble] reset', ensembler.seg_countmap_4d)
        return det_result_batch, seg_result_batch

    def inference(self, imgs, img_metas, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W, D).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            det_results: list[tuple[bbox_nx7, label_nx1], tuple, ...]
            seg_results: Tensor, The output segmentation map.

        """
        assert imgs.shape[0] == 1 and len(img_metas) == 1

        assert self.test_cfg.mode in ['slide', 'whole']

        # print(f'INFER: Flip {flip} Direction {flip_direction} dims {flip_dims}')
        if self.test_cfg.mode == 'slide':
            det_results, seg_results = self.sliding_window_inference(imgs, img_metas, mode = self.blend_mode, 
                                                                    padding_mode= self.padding_mode,
                                                                    sigma_scale=self.sigma_scale, 
                                                                    rescale= rescale)
        else:
            # NOTE: whole inference should be implemented
            det_results, seg_results= self.whole_inference(imgs, img_metas, rescale)

        seg_results = torch.cat([torch.softmax(seg, dim=1) if seg.shape[1] > 1 else torch.sigmoid(seg)
                                for seg in seg_results], axis = 0).cpu()
        return det_results, seg_results 


    def simple_test_global(self, img, img_meta, rescale=True, need_probs = False):
        """Simple test with single image."""
        det_results, seg_results = self.inference(img, img_meta, rescale)
        # print_tensor('simple test %s' %need_logits, seg_logit)
        if need_probs:
            seg_pred = seg_results.cpu() #.float().cpu().numpy()
        else:
            seg_pred = seg_results.argmax(dim=1) if seg_results.shape[1] > 1 else seg_results.squeeze(1) > 0.5
            # print_tensor(f'[SimpleTest] pred unique labels {torch.unique(seg_pred)}', seg_pred)
            seg_pred = seg_pred.to(torch.uint8).cpu()
            # unravel batch dim
            seg_pred = list(seg_pred)
        
        # pdb.set_trace()
        bbox_results = [bbox2result3d(det_bboxes, det_labels, self.bbox_head.num_classes) 
                        for det_bboxes, det_labels in det_results]
         # most outer image level; next inner class level; most inner 
        # pdb.set_trace()
        return bbox_results, seg_pred

    def aug_test_global(self, imgs, img_metas, rescale=True, need_probs = False):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        # print(img_metas)
        det_resutls, seg_results = self.inference(imgs[0], img_metas[0], rescale)
        imgs[0] = None; torch.cuda.empty_cache()
        infer_times = 1
        # print('aug, post inference', seg_logit.shape)
        for i in range(1, len(imgs)):
            cur_det_results, cur_seg_results = self.inference(imgs[i], img_metas[i], rescale)
            # Cumulvate Moving Average: CMA_n+1 = (X_n+1 + n * CMA_n)/ (n + 1); CMA_n = (x1 + x2 + ...) / n
            seg_results = (cur_seg_results + seg_results * infer_times) / (infer_times + 1)
            infer_times += 1
            det_resutls.extend(cur_det_results)
            imgs[i] = None; torch.cuda.empty_cache()

        if need_probs:
            seg_pred = seg_results.cpu() #.float().cpu().numpy()
        else:
            seg_pred = seg_results.argmax(dim=1) if seg_results.shape[1] > 1 else seg_results.squeeze(1) > 0.5
            # print_tensor(f'[SimpleTest] pred unique labels {torch.unique(seg_pred)}', seg_pred)
            seg_pred = seg_pred.to(torch.uint8).cpu()
            # unravel batch dim
            seg_pred = list(seg_pred)

        # detection
        bbox_nx7 = torch.cat([d1[0] for d1 in det_resutls], axis = 0) # nx7
        label_nx1 = torch.cat([d1[1] for d1 in det_resutls], axis = 0) # nx1
        if bbox_nx7.shape[0] < 1: 
            return [[torch.zeros((0, 7)), ]], None
            
        dets_clean, keep_idx = batched_nms_3d(bbox_nx7[:, :6], bbox_nx7[:, 6], 
                                             label_nx1, self.test_cfg['nms'])
        # TODO: 2897118_image.nii cause exception,  max_coordinate = boxes.max()  operation does not have an identity
        label_nx1 = label_nx1[keep_idx]

        if self.test_cfg['max_per_img'] > 0:
            dets_clean = dets_clean[:self.test_cfg['max_per_img']]
            label_nx1 = label_nx1[:self.test_cfg['max_per_img']]
        
        bbox_results = [bbox2result3d(dets_clean, label_nx1, self.bbox_head.num_classes) ]
        return bbox_results, seg_pred

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                            f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-3:])
        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test_global(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                        'inference with batch size ' \
                                        f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test_global(imgs, img_metas, **kwargs)

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        preds, *aux_output = self.encode_decode(img, img_meta)
        img_meta_dict = get_meta_dict(img_meta)
        # print(img_meta[0])
        target_shape = img_meta_dict['spatial_shape'] # TODO: determine the key word in image_meta_dict for original shape
        if img_meta_dict.get('new_pixdim', False):
            if rescale and (preds.shape[-3:] != target_shape):    
                preds = resize_3d(
                    preds,
                    size=target_shape,
                    mode=self.get_output_mode(), #'bilinear'
                    align_corners=self.align_corners,
                    warning=False)
        return preds

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W, D).
            img_metas (list[dict]): List of image information. 
            e.g.[{'img_shape': (H, W, D), 'scale_factor': 1}]
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result3d(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        if self.with_seghead: 
            mask = self.seg_head.simple_test(feat, img_metas, rescale = rescale)
        else: mask = None

        return bbox_results, mask

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxWxD,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test( # dense_test_mixins
            feats, img_metas, rescale=rescale) 
        bbox_results = [
            bbox2result3d(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        if self.with_seghead: 
            mask = self.seg_head.aug_test(feats, img_metas, rescale = rescale)
        else: mask = None

        return bbox_results, mask

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels


get_meta_dict  = lambda img_meta: img_meta[0]['img_meta_dict'] if isinstance(img_meta, (list, tuple)) else img_meta['img_meta_dict']

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
