from ..backbones.hrnet_3d import *
from ..decode_heads.fcn_head_3d import *


# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='SyncBN', requires_grad=True) #Sync

backbone_kargs = dict(pretrained2d = True, 
        conv_cfg = conv_cfg, 
        norm_cfg=norm_cfg,
        norm_eval=False,
        conv1_stride_t=2,
        conv2_stride_t=1,
        reduce_conv3d4z = False,
        in_channels=1,
        stem_channel = 32, 
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(32, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=3,
                num_branches=3,
                block='BASIC',
                num_blocks=(2, 2, 2),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='BASIC',
                num_blocks=(2, 2, 2, 2),
                num_channels=(18, 36, 72, 144))))

head_kargs = dict(
        # type='FCNHead3D',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=128,
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=1,
        conv_cfg = conv_cfg, 
        norm_cfg=norm_cfg,
        align_corners=False,
        # parameters for implicit semantic data augmentation
        # is_use_isda = True, 
        # isda_lambda = 2.5,
        # start_iters = 1,
        # max_iters = 4e5,
        # OHEM
        # sampler=dict(type='OHEMPixelSampler', min_kept=600000),
        # loss_decode =dict(
        #             # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        #             # type='ComboLoss', use_sigmoid=True, loss_weight=(1.0, 0.5), pos_topk = 1024,
        #             type='ComboLossMed', loss_weight=(1.0, 0.5), num_classes = 1, 
        #             class_weight = (0.8, 1.0, 1.1, 1.0, 0.8, 0.8),  verbose = False,
        #             group_dict = {0: [0], 1: [1,2,3], 2:[4,5]},  group_loss_weight = 0.5,
        #             focal_loss_gamma = 2.0
        #             ),
            )

class FCN_HRnet3d(nn.Module):
    
    backbone_kargs = backbone_kargs
    head_kargs = head_kargs
    def __init__(self, 
                # backbone_kargs, 
                # head_kargs,
                in_channels=1,
                out_channels=3,
                neck_kargs = None, 
                auxiliary_kargs = None,
                 ):
        """
        
        args:
            in_channels: the channel num of input image
            out_channels: the num of classes to be predicted
        """
        super().__init__()
        backbone_kargs['in_channels'] = in_channels
        head_kargs['num_classes'] =  out_channels
        # head_kargs['']
        self.backbone = HRNet3D(**self.backbone_kargs)
        self.head = FCNHead3D(**self.head_kargs)
        # if neck_kargs is not None:
        #     self.neck = self.
        
    def forward(self, x):
        spatial_shape = x.shape[2:]
        feat_list = self.backbone(x)
        output = self.head(feat_list)
        output = resize_3d(output, spatial_shape, mode= 'trilinear')
        return output