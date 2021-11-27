_base_ = [
    '../datasets/allct_bone_160x192x128.py',
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
    type='DenseCL3D',
    backbone=dict(
        type='SwinTransformer3D',
        embed_dims= stem_channels * 2,
        in_channels=1,
        depths=[2, 2, 4],
        strides=(4, 2, 2),
        num_heads=[4, 8, 16],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(2, ),
        with_cp=False,
        ),
    neck=dict(
        type='DenseCLNeck3D',
        in_channels = stem_channels * (2**3), 
        hid_channels = stem_channels * (2**3),
        out_channels = 64, 
        num_grid=None,
        ),
    head=dict(type='ContrastiveHead', temperature=0.2), 
    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    # queue_len=96*96*96,
    feat_dim=64,
    momentum=0.999,
    loss_lambda=0.5,
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)