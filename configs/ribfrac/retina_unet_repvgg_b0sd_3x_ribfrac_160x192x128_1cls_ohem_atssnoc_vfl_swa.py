_base_ = [
    '../ribfrac/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/models_med/retina_unet_repvgg_b0sd_atssnoc_160x192x128.py',
    # '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py', 
     '../_base_/swa.py'
]

data = dict(train=dict(sample_rate = 1.0), 
            val=dict(sample_rate = 1.0), 
            test= dict(sample_rate = 0.1))
            
model = dict(bbox_head = dict(anchor_generator = dict(verbose = False), 
                               loss_cls = dict(verbose = False), 
                                verbose = False), 
           )

swa_training = True
swa_load_from = 'work_dirs/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl/best_recall@8@0.1_epoch_12.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001, 
                # _delete_ = True, type='AdamW', lr=0.0001, weight_decay=0.0001, 
                paramwise_cfg = dict(custom_keys={
                                    r'stage\d.\d?.?rbr': dict(decay_mult = 0.0)
                                        # '.rbr_dense': dict(decay_mult=0), 
                                        # '.rbr_1x1': dict(decay_mult=0), 
                                        # '.bias': dict(decay_mult = 0), 
                                        # '.bn': dict(decay_mult = 0)
                                    },  #stage\d.\d.rbr
                ),
        ) 
# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12, #24
    step_ratio_up=0.0)
swa_runner = dict(type='EpochBasedRunner', max_epochs=24)  
swa_interval = 2

# swa_optimizer_config
swa_optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 8, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G

# yapf:disable
log_config = dict(interval=10, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

swa_checkpoint_config = dict(interval=4, filename_tmpl='swa_epoch_{}.pth', max_keep_ckpts = 4)
evaluation=dict(interval=2, start=None, 
                save_best='recall@8@0.1', rule = 'greater', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29346 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_swa.py 3 --gpus 3 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798