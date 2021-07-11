import warnings

import torch
from mmdet.core import bbox2result3d
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base3d import BaseDetector3D
from ...datasets.pipelines import Compose


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
                        instance_key="gt_instance_seg",
                        save_key="present_instances"), 
                    dict(type = 'Instances2Boxes', 
                        instance_key="target",
                        map_key="instance_mapping",
                        box_key="gt_bboxes",
                        class_key="gt_labels",
                        present_instances="present_instances"),
                    dict(type = 'Instances2SemanticSeg', 
                        map_key="instance_mapping",
                        seg_key = 'gt_semantic_seg',
                        present_instances="present_instances",
                        )]
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
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.seg_head = build_head(seg_head) if seg_head is not None else None

        self.gpu_pipelines = Compose(gpu_aug_pipelines + mask2bbox_cfg) if mask2bbox_cfg is not None else None

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

    def update_img_metas(self, imgs, img_metas, gt_instance_seg, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape # TODO, what?
             
        gt_keys = ['img', 'gt_bboxes', 'gt_labels', 'gt_semantic_seg']
        # print_tensor('rawgt', gt_semantic_seg) # {key: [meta1, meta],}
        with torch.no_grad(): 
            img_meta_dict = [a['img_meta_dict'] for a in img_metas]
            gt_meta_dict = [a['gt_instance_seg_meta_dict']for a in img_metas]
            data_dict = self.gpu_pipelines({'img': imgs, 
                                            'gt_instance_seg': gt_instance_seg, 
                                            'img_meta_dict' : img_meta_dict, 
                                            'gt_instance_seg_meta_dict' : gt_meta_dict})
        return [data_dict[k] for k in gt_keys]
        
    @property
    def with_seghead(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_train(self,
                      img,
                      img_metas,
                      gt_instance_seg,
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
                                            img, img_metas, gt_instance_seg)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.with_seghead: 
            loss_seg = self.seg_head.forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_seg)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W, D).
            img_metas (list[dict]): List of image information. 
            e.g.[{'img_shape': (32, 32, 3), 'scale_factor': 1}]
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
