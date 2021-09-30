# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class VFNet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)


from .single_stage_3d import SingleStageDetector3D

@DETECTORS.register_module()
class VFNet3D(SingleStageDetector3D):
    """Implementation of `VarifocalNet3D
        (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

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
        super(VFNet3D, self).__init__(backbone, neck, bbox_head, seg_head, 
                                          train_cfg, test_cfg, pretrained, init_cfg, 
                                          **kwargs)