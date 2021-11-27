_base_ = [
    '../_base_/models_med/retina_vfnet_r34_4l16c_atssnoc_160x192x128_3cls_3level.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/swa.py',
]

pretrained = 'work_dirs/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'

img_dir = 'data/Task113_RibFrac_KYRe'
key2suffix = {'img_fp': '_image.nii',  'seg_fp': '_instance.nii', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 2, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_1105.json', img_dir=img_dir, key2suffix = key2suffix), 
            val=dict(sample_rate = 1.0, json_filename = 'dataset_1105.json', img_dir=img_dir, key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_1105.json',img_dir=img_dir, key2suffix = key2suffix))
            
model = dict(backbone = dict(verbose = False, 
                            with_cp = False, 
                            init_cfg = dict(type = 'Pretrained', checkpoint= pretrained)), 
            seg_head = dict(verbose = False),
            neck = dict(verbose = False),
            bbox_head = dict(verbose = False, use_vfl = True, 
                            anchor_generator = dict(verbose = False), 
                            # loss_cls = dict(verbose = False), 
                             ))

find_unused_parameters=True
load_from = 'work_dirs/retina_vfnet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_refine/latest.pth'
resume_from =  None #'work_dirs/retina_vfnet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc/latest.pth' 

# optimizer
optimizer = dict(#type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=1e-4, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                policy='CosineAnnealing',  min_lr=1e-5,
                # policy='poly', power=0.99, min_lr=1e-5, 
                by_epoch=False, warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=4, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=30, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))


# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/ribfrac/retina_vfnet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_refine.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29239 bash ./tools/dist_train.sh configs/ribfrac/retina_vfnet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_refine.py 3 --gpus 3 #--no-validate


# swa_training = True
# swa_load_from = 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl/latest.pth'
# swa_resume_from = None
# # swa optimizer
# swa_optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
# # swa learning policy
# swa_lr_config = dict(
#     policy='cyclic',
#     target_ratio=(1, 0.01),
#     cyclic_times=12,
#     step_ratio_up=0.0)
# swa_runner = dict(type='EpochBasedRunner', max_epochs=12)  
# swa_interval = 2
# # swa_optimizer_config
# swa_optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 8, norm_type = 2)) # 31G
# swa_checkpoint_config = dict(interval=4, filename_tmpl='swa_epoch_{}.pth', max_keep_ckpts = 4)

#  sw_batch_size = 1, overlap = 0.25, runtime = 19.1s
#  sw_batch_size = 2, overlap = 0.25, runtime = 16.7s
#  sw_batch_size = 2, overlap = 0.33, runtime = 18.3s
#   sw_batch_size = 2, overlap = 0.5, runtime = 36.5