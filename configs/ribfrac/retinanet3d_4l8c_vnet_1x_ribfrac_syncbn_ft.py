_base_ = [
    '../_base_/models/retinanet_r18_4l8c_vnet_3d_syncbn.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

find_unused_parameters=True
load_from = None # 'work_dirs/retinanet3d_4l8c_vnet_1x_ribfrac_syncbn/latest.pth'
resume_from = None #'work_dirs/retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft/latest.pth' 

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, type='Fp16OptimizerHook', loss_scale=512.,
                        grad_clip = dict(max_norm = 16, norm_type = 2)
                        ) #only reduce from 10G->7G

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=20, hooks=[#dict(type='TextLoggerHook'), 
                dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(iou_thr=[0.2], proposal_nums=(10, 50, 100))
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft.py --no-validate
# CUDA_VISIBLE_DEVICES=2,4,5 PORT=29001 bash ./tools/dist_train.sh configs/ribfrac/retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft.py 3 --no-validate