_base_ = [
    '../_base_/models_med/retina_unet_r34_4l16c_atssnoc_160x192x128_vfl_3cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/swa.py',
]

data = dict(train=dict(sample_rate = 1.0), 
            val=dict(sample_rate = 0.5), 
            test= dict(sample_rate = 0.1))
            
model = dict(bbox_head = dict(
                            anchor_generator = dict(verbose = False), 
                            loss_cls = dict(verbose = False), 
                            verbose = False, 
                            use_vfl = True))

find_unused_parameters=True
load_from = 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls/best_mAP_epoch_12_round2.pth'
resume_from =  None #'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls/latest.pth' 

# optimizer
optimizer = dict(#type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=0.0005, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=5e-5, by_epoch=False,
                #  warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=10, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, start=4, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                # save_best='mAP@8@0.1', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))


# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls.py 
# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29034 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls.py 3 --gpus 3 #--no-validate


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