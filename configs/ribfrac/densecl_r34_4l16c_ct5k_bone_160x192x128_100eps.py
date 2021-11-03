_base_ = [
    '../_base_/models_med/densecl_r18_4l16c_160x192x128.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


data = dict(samples_per_gpu = 4, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'image_nii_fps_1025.txt', split='train'), 
            val=dict(sample_rate = 1.0, json_filename = 'image_nii_fps.txt', split='test'), 
            test= dict(sample_rate = 0.1, json_filename = 'image_nii_fps.txt', split='test')
            )
            

            
model = dict(
        backbone = dict(depth = '343d'), 
        queue_len= 3 * 2**14)

find_unused_parameters=True
load_from = 'work_dirs/densecl_r18_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'
resume_from = None  #'work_dirs/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'

# optimizer
optimizer = dict(
                type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, 
                # _delete_ = True, type='AdamW', lr=0.0002, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=1e-4, by_epoch=False, 
                 warmup='linear', warmup_iters=500, 
                #  policy='CosineAnnealing', min_lr=0.
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=30, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

# evaluation=dict(interval=4, start=4, metric='mAP', 
#                 save_best = 'mAP', rule = 'greater', 
#                 # save_best='mAP@8@0.1', 
#                 iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))

# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps.py --no-validate
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29010 bash ./tools/dist_train.sh configs/ribfrac/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps.py 3 --gpus 3 --no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798


#  sw_batch_size = 1, overlap = 0.25, runtime = 19.1s
#  sw_batch_size = 2, overlap = 0.25, runtime = 16.7s
#  sw_batch_size = 2, overlap = 0.33, runtime = 18.3s
#   sw_batch_size = 2, overlap = 0.5, runtime = 36.5