_base_ = [
    '../datasets/allct_bone_160x160x128.py',
]
num_classes = 1 # this is an RPN 
# model settings
conv_cfg = dict(type = 'Conv3d')
norm4head = dict(type='GN', num_groups=8, requires_grad=True) 
norm_cfg = dict(type='IN3d', requires_grad=True) 
# bs=2, ng=8, chn=2, m=18.1G;  bs=2, ng=2, m=17.2G;  bs=2, ng=8, chn=1, m=19.3G 
stem_channels = 16
fpn_channel = stem_channels * (2**3)
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SwinTransformer3D4SimMIM',
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
        in_channels=[stem_channels * (2**(c+ 1) ) for c in range(4)], 
        fixed_out_channels = fpn_channel,
        start_level=0,
        conv_cfg = conv_cfg, 
        norm_cfg = norm_cfg, 
        add_extra_convs=False,
        num_outs=4, 
        upsample_cfg=dict(type='deconv3d', mode=None, use_norm = True, 
                    kernel_size = (2,2,2), stride = (2,2,2)),
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
        num_classes= 1,
        conv_cfg = conv_cfg, 
        norm_cfg=norm4head,
        align_corners=False,
        gt_index = 0),

    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    mask_cfg = dict(input_size=(160, 160, 128), mask_patch_size=32,
                                 stem_stride=4, mask_ratio=0.6), 
                                 
    test_cfg=dict(roi_size = {{ _base_.patch_size }}, sw_batch_size = 6,
        blend_mode = 'constant' , overlap=0.05, sigma_scale = 0.125, # 'gaussian or constant
        padding_mode='constant' )

)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)