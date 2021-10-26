_base_ = [
    '../_base_/models_med/retina_unet_r34_4l8c_atss_160x192x128.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(train=dict(sample_rate = 1.0), 
            val=dict(sample_rate = 0.4), 
            test= dict(sample_rate = 0.1))

find_unused_parameters=True
load_from = 'work_dirs//latest.pth'
resume_from = None #'work_dirs/retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_rf/latest.pth' 

# optimizer
optimizer = dict(#type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=0.0002, weight_decay=0.0005
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
# fp16 = dict(loss_scale=512.) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=5e-5, by_epoch=False,
                 warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=64)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=20, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, iou_thr=[0.1], proposal_nums=(1, 2, 8, 50))
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_ft.py 
# CUDA_VISIBLE_DEVICES=1,3 PORT=29022 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_ft.py 2 --gpus 2 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recallretina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_rf