_base_ = [
    '../datasets/ribfrac_instance_semantic.py',
]

# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='SyncBN', requires_grad=True) #Sync
stem_channels = 16
fpn_channel = stem_channels * (2**3)
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
        num_stages=5,
        strides=(2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1),
        out_indices=(1, 2, 3, 4, 5, 6), # 0 is input image 
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
    neck=dict(
        type='FPN3D',
        in_channels=[stem_channels * (2**c) for c in range(6)],
        fixed_out_channels = fpn_channel,
        start_level=2,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        add_extra_convs=False,
        num_outs=6),
    bbox_head=dict(
        type='RetinaHead3D', 
        num_classes=2,
        in_channels=fpn_channel,
        stacked_convs=4,
        feat_channels=fpn_channel,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        anchor_generator=dict( 
            type='AnchorGenerator3D', 
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder3D',
            target_means=[.0, .0, .0, .0, 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
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
        num_classes=2,
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
        loss_decode =dict(
                    type='ComboLossMed', loss_weight=(1.0, 0.66), 
                    num_classes = 2, 
                    class_weight = (0.33, 1.0),  verbose = False,  
                    dice_cfg = dict(ignore_0 = True),  #
                    # focal_loss_gamma = 1.0
                    ),
            ),
    # convert instance mask to bbox 
    mask2bbox_cfg = [dict(type = 'FindInstances', 
                        instance_key="seg",
                        save_key="present_instances"), 
                    dict(type = 'Instances2Boxes', 
                        instance_key="seg",
                        map_key="instance_mapping",
                        box_key="gt_bboxes",
                        class_key="gt_labels",
                        present_instances="present_instances"),
                    dict(type = 'Instances2SemanticSeg', 
                        instance_key = 'seg',
                        map_key="instance_mapping",
                        seg_key = 'seg',
                        present_instances="present_instances",
                        )],
    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
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
        nms=dict(type='nms', iou_threshold=0.25), # TODO: nms 3D
        max_per_img=100)
    
    )
