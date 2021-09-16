# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_3d import SingleStageDetector3D


@DETECTORS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg, 
                                        test_cfg, pretrained, init_cfg)



@DETECTORS.register_module()
class RetinaNet3D(SingleStageDetector3D):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 seg_head = None, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None, 
                 **kwargs,
                 ):
        super(RetinaNet3D, self).__init__(backbone, neck, bbox_head, seg_head, 
                                          train_cfg, test_cfg, pretrained, init_cfg, 
                                          **kwargs)