_base_ = [
    '../datasets/ribfrac_instance_semantic_inhouse_160x192x128_3cls.py',
]
num_classes = 3 # this is an RPN 
# model settings
conv_cfg = dict(type = 'Conv3d')
norm4head = dict(type='GN', num_groups=8, requires_grad=True) 
norm_cfg = dict(type='IN3d', requires_grad=True) 
# bs=2, ng=8, chn=2, m=18.1G;  bs=2, ng=2, m=17.2G;  bs=2, ng=8, chn=1, m=19.3G 
stem_channels = 16
fpn_channel = stem_channels * (2**3)
model = dict(
    type='RetinaNet3D',
    backbone=dict(
        type='ResNet3dIso', verbose = False, 
        deep_stem = True,
        avg_down=True,
        depth='343d', # 18.3G 
        in_channels=1,
        stem_stride_1 = 2,
        stem_stride_2 = 1, 
        stem_channel_div = 1, 
        stem_channels= stem_channels, # 32 
        base_channels= stem_channels * 2, # 64 
        num_stages=4,
        strides=(2, 2, 2, 2), # 64, 128, 256, 320
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3, 4, 5), # 0 is input image 
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        style='pytorch',
        # non_local=((0, 0), (0, 0), (0, 0), (0, 0)),
        # non_local_cfg=dict(
        #     conv_cfg = conv_cfg,
        #     norm_cfg=norm_cfg,
        #     out_projection = True,
        #     reduction=8)
        ),
    neck=dict(
        type='FPN3D',
        in_channels=[stem_channels * (2**c) for c in range(5)], 
        fixed_out_channels = fpn_channel,
        start_level=1,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        add_extra_convs=False,
        num_outs=5),
    bbox_head=dict(
        type='VFNetHead3D', #verbose = True, 
        num_classes=num_classes,
        in_channels=fpn_channel,
        stacked_convs=3,
        start_level = 1, 
        feat_channels=fpn_channel,
        conv_cfg = conv_cfg, 
        norm_cfg = norm4head, 
        #NOTE: stride == base_size the len of stride should be identical to fpn levels
        strides=(4, 8, 16, 32),
        anchor_generator=dict( 
            type='AnchorGenerator3D', #verbose = True, 
            octave_base_scale=2,
            scales_per_octave=1, 
            ratios=[1.0], center_offset=0.0,
            strides=(4, 8, 16, 32)),
        # bbox_coder=dict(
        #     type='DeltaXYWHBBoxCoder3D',
        #     target_means=[.0, .0, .0, .0, 0., 0.],
        #     target_stds=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], 
        #     clip_border=False),
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True, 
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=8.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=0.2),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=0.4)
        ), 

    seg_head = dict(
        type='FCNHead3D', # verbose = True, 
        in_channels= stem_channels * 2,
        in_index=0,
        channels= stem_channels,
        # input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes + 1,
        conv_cfg = conv_cfg, 
        norm_cfg=norm4head,
        align_corners=False,
        gt_index = 0, 
        # parameters for implicit semantic data augmentation
        # is_use_isda = True, 
        # isda_lambda = 2.5,
        # start_iters = 1,
        # max_iters = 4e5,
        loss_decode =dict(
                    type='ComboLossMed', loss_weight=(1.0 * 0.4, 0.66 * 0.4), 
                    num_classes = num_classes + 1, class_weight = (0.33, 1.5, 1.0, 1.0),  verbose = False,   #(0.33, 1.0)
                    dice_cfg = dict(ignore_0 = True, verbose = False) # act = 'sigmoid',
                    ),
            ),
    # convert instance mask to bbox 
    mask2bbox_cfg = [dict(type = 'FindInstances', #verbose = True, 
                        instance_key="seg",
                        save_key="present_instances"), 
                    dict(type = 'Instances2Boxes', #verbose = True, 
                        instance_key="seg",
                        map_key="inst2cls_map", 
                        box_key="gt_bboxes",
                        class_key="gt_labels", move_jitter = 2, 
                        present_instances="present_instances"),
                    dict(type = 'Instances2SemanticSeg', #verbose = True, 
                        instance_key = 'seg',
                        map_key="inst2cls_map",
                        seg_key = 'seg', add_background = True, 
                        present_instances="present_instances"), 
                    # dict(type = 'SaveImaged', keys = ('img', 'seg'), 
                        # output_dir = f'work_dirs/retinanet3d_4l8c_vnet_1x_ribfrac_1cls/debug', 
                        # resample = False, save_batch = True, on_gpu = True),  
                    ],
    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='ATSSAssigner3D', # could be replaced by ATSSAssigner
            topk = 9, center_within = False, 
            ignore_iof_thr=-1, verbose = False,
            iou_calculator=dict(type='BboxOverlaps3D')
            ),
        sampler=dict(
                type='HardNegPoolSampler',
                num=64, pool_size = 32,
                pos_fraction=0.33,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
        allowed_border=3,
        pos_weight=1.0,
        debug=False),
    test_cfg=dict(
        nms_pre=200,
        # nms_pre_tiles = 1000, 
        min_bbox_size=2,
        score_thr=0.15,
        nms=dict(type='nms', iou_threshold=0.1), # 
        # https://github.com/MIC-DKFZ/nnDetection/blob/7246044d8824f7b3f6c243db054b61420212ad05/nndet/ptmodule/retinaunet/base.py#L419
        max_per_img=32, 
        mode='slide', roi_size = {{ _base_.patch_size }}, sw_batch_size = 2,
        blend_mode = 'gaussian' , overlap=0.5, sigma_scale = 0.125, # 'gaussian or constant
        padding_mode='constant' )
)

    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)