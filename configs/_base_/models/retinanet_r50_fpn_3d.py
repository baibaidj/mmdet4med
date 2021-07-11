# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='SyncBN', requires_grad=True) #Sync
stem_channels = 8
num_classes = 1
model = dict(
    type='RetinaNet3D',
    backbone=dict(
        type='ResNet3dIso', # 
        deep_stem = True,
        avg_down=True,
        depth='183d',
        in_channels=1,
        stem_stride_1 = 1,
        stem_stride_2 = 1, 
        stem_channels= stem_channels,
        base_channels= stem_channels * 2,
        num_stages=4,
        strides=(2,2,2,2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3, 4, 5),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        style='pytorch',
        verbose = False, 
        # non_local=((0, 0), (0, 0), (0, 0), (0, 0)),
        # non_local_cfg=dict(
        #     conv_cfg = conv_cfg,
        #     norm_cfg=norm_cfg,
        #     out_projection = True,
        #     reduction=8)
        ),
    neck= dict(
        type = 'UnetNeck3D',
        in_channels=[ stem_channels * (2**c) for c in range(5)],
        channels=[ stem_channels * (2**c) for c in range(4, 0, -1)],  # 128 
        norm_cfg=norm_cfg,
        conv_cfg= conv_cfg,
        align_corners=False,
        skip_levels = 4, 
        sece_blocks = [1, 1, 1],  
        is_p3d = False,
        nlblocks = (0, 0, 0), 
        nl_cfg = dict(type = 'cca',
                    fusion_type = 'add', 
                    key_source = 'self', 
                    psp_size = None,
                    reduction = 4, out_projection = False) # out_projection not good
    ),
    bbox_head=dict(
        type='RetinaHead3D',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict( 
            type='AnchorGenerator3D', 
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder3D',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    seg_head = dict(
        type='FCNHead3D',
        in_channels= stem_channels * 2,
        in_index=0,
        channels= stem_channels * 2,
        # input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        conv_cfg = conv_cfg, 
        norm_cfg=norm_cfg,
        align_corners=False,
        gt_index = 0, 
        verbose = False, 
        # parameters for implicit semantic data augmentation
        # is_use_isda = True, 
        # isda_lambda = 2.5,
        # start_iters = 1,
        # max_iters = 4e5,
        # OHEM
        # sampler=dict(type='OHEMPixelSampler', min_kept=600000),

        # loss_decode =dict(
        #             type='ComboLossWHD', loss_weight=(1.0, 0.0015), 
        #             num_classes = num_classes, gt_index4_ce = 0, 
        #             class_weight = (0.2, 1.0, 1.0),  verbose = False,  
        #             whd_cfg =dict( gm_alpha=-3,  verbose = False, 
        #                             upper_bound_neg = None, upper_bound_pos = 9e4,
        #                             sample_ratio_neg = 0, sample_ratio_pos = 0.85, 
        #                             is_correct_cb_inc = True, act_cfg = 'softmax')
        #             )
        loss_decode =dict(
                    type='ComboLossMed', loss_weight=(1.0, 0.66), 
                    num_classes = num_classes, 
                    class_weight = (0.33, 1.0),  verbose = False,  
                    dice_cfg = dict(ignore_0 = True),  #
                    # focal_loss_gamma = 1.0
                    ),
            ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps3D')
            ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5), # TODO: nms 3D
        max_per_img=100)
    
    )
