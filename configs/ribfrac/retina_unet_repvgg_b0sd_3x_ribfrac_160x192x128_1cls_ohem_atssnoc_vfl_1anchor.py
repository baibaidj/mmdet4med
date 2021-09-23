_base_ = [
    '../_base_/models/retina_unet_repvgg_b0sd_atssnoc_160x192x128_vfl_syncbn.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(train=dict(sample_rate = 1.0), 
            val=dict(sample_rate = 0.5), 
            test= dict(sample_rate = 0.1))
            
model = dict(bbox_head = dict(
                            anchor_generator = dict(verbose = False), 
                            loss_cls = dict(verbose = False), 
                            verbose = False, 
                            use_vfl = True, ))


find_unused_parameters=True
load_from = 'work_dirs/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor/best_recall@8@0.1_epoch_8_bak.pth'
resume_from = None #'work_dirs/retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_rf/best_recall@8@0.1_epoch_12.pth' 

# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001, 
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
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2), 
                        use_repvgg = True) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy # 'poly'  power=0.99, 
# lr_config = dict(_delete_=True, policy='CosineAnnealing',min_lr=1e-4, by_epoch=False,
#                 #  warmup='linear', warmup_iters=400, 
#                  )
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12, #24
    step_ratio_up=0.0
    )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=10, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=4, 
                save_best='recall@8@0.1', rule = 'greater', 
                iou_thr=[0.1],  proposal_nums=(1, 2, 4, 8, 50))
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/ribfrac/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29346 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor.py 3 --gpus 3 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798