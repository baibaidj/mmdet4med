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
        type='SwinTransformer3D',
        embed_dims= stem_channels * 2,
        in_channels=1,
        depths=[2, 2, 4, 2],
        strides=(4, 2, 2, 2),
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        ),
    neck=dict(
        type='FPN3D',
        in_channels=[stem_channels * 2 * (2**c) for c in range(4)], 
        fixed_out_channels = fpn_channel,
        start_level=0,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        add_extra_convs=False,
        num_outs=4),
    bbox_head=dict(
        type='ATSSHead3DNOC', #verbose = True, 
        num_classes=num_classes,
        in_channels=fpn_channel,
        stacked_convs=4,
        start_level = 0, 
        feat_channels=fpn_channel,
        conv_cfg = conv_cfg, 
        norm_cfg = norm4head, 
        anchor_generator=dict( 
            type='AnchorGenerator3D', #verbose = True, 
            octave_base_scale=2,
            scales_per_octave=2, 
            ratios=[1.0],
            strides=[4, 8, 16, 32]), #NOTE: stride == base_size the len of stride should be identical to fpn levels
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder3D',
            target_means=[.0, .0, .0, .0, 0., 0.],
            target_stds=[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], 
            clip_border=False),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25, 
            loss_weight=1.0),
        use_vfl=False, 
        loss_cls_vfl=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=8.0),
        loss_bbox=dict(type='GIoULoss3D', loss_weight=0.4), 
        # loss_centerness=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.66)
        ), 

    seg_head = dict(
        type='FCNHead3D', # verbose = True, 
        in_channels= fpn_channel,
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
                    type='ComboLossMed', loss_weight=(1.0 * 0.2, 0.66 * 0.2), 
                    num_classes = num_classes + 1, class_weight = (0.33, 1.25, 1.0, 1.0),  verbose = False,   #(0.33, 1.0)
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
            type='ATSSAssigner3D',
            topk = 4, center_within = False, 
            ignore_iof_thr=-1, verbose = False,
            iou_calculator=dict(type='BboxOverlaps3D')
            ),
        sampler=dict(
                type='HardNegPoolSampler',
                num=32, pool_size = 20,
                pos_fraction=0.4,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
        allowed_border=3,
        pos_weight=1.0,
        debug=False),
    test_cfg=dict(
        nms_pre=200,
        # nms_pre_tiles = 1000, 
        min_bbox_size=2,
        score_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.1), # 
        # https://github.com/MIC-DKFZ/nnDetection/blob/7246044d8824f7b3f6c243db054b61420212ad05/nndet/ptmodule/retinaunet/base.py#L419
        max_per_img=32, 
        mode='slide', roi_size = {{ _base_.patch_size }}, sw_batch_size = 6,
        blend_mode = 'gaussian' , overlap=0.5, sigma_scale = 0.125, # 'gaussian or constant
        padding_mode='constant' )
)

    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)