_base_ = [
    '../_base_/models/retinanet_r18_4l8c_vnet_3d.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.0, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, type='Fp16OptimizerHook', loss_scale=512., distributed = False,
                        grad_clip = dict(max_norm = 16, norm_type = 2)
                        ) #only reduce from 10G->7G

load_from = None
resume_from = None
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(interval=50)
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/retinanet3d_4l8c_vnet_1x_ribfrac.py --no-validate
# CUDA_VISIBLE_DEVICES=0,2,5 PORT=29103 bash ./tools/dist_train.sh configs/ribfrac/retinanet3d_4l8c_vnet_1x_ribfrac.py 3 --no-validate