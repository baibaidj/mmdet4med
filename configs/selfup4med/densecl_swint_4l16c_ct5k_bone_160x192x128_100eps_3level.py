_base_ = [
    '../_base_/models_med/densecl_swint_4l16c_160x192x128.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


data = dict(samples_per_gpu = 2, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'image_nii_fps_1025.txt', split='train'), 
            val=dict(sample_rate = 1.0, json_filename = 'image_nii_fps.txt', split='test'), 
            test= dict(sample_rate = 0.1, json_filename = 'image_nii_fps.txt', split='test')
            )
            
model = dict(backbone=dict(depths=[2, 2, 4]), 
             queue_len= 3 * 2**14, 
             init_cfg = None, #init_cfg=dict(type='Pretrained', checkpoint=pretrained)
             )

find_unused_parameters=True
load_from = 'work_dirs/densecl_swint_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'
resume_from = None #'work_dirs/densecl_swint_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'

# optimizer
optimizer = dict(_delete_ = True,
                type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001, 
                # _delete_ = True, type='AdamW', lr=0.0002, weight_decay=0.0001
                # paramwise_cfg=dict(
                #         custom_keys={
                #         r'absolute_pos_embed': dict(decay_mult=0.), #r'stage\d.\d?.?rbr'
                #         r'relative_position_bias_table': dict(decay_mult=0.), # backbone.stages.1.blocks.1.attn.w_msa.relative_position_bias_table
                #         r'norm': dict(decay_mult=0.) #backbone.stages.\d.downsample.norm
                #         }
                #         )
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 64, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=1e-4, by_epoch=False, 
                 warmup='linear', warmup_iters=1000, 
                #  policy='CosineAnnealing', min_lr=0z.
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=20, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/densecl_swint_4l16c_ct5k_bone_160x192x128_100eps.py --no-validate
# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29227 bash ./tools/dist_train.sh configs/ribfrac/densecl_swint_4l16c_ct5k_bone_160x192x128_100eps_3level.py 3 --gpus 3 --no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798