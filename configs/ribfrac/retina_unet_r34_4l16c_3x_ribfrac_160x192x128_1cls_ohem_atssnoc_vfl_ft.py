_base_ = [
    '../_base_/models_med/retina_unet_r34_4l16c_atssnoc_160x192x128_vfl_1cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/swa.py',
]

img_dir = 'data/Task113_RibFrac_KYRe'
key2suffix = {'img_fp': '_image.nii',  'seg_fp': '_instance.nii', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 3, workers_per_gpu= 9, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_1130_truesplit.json', img_dir=img_dir, key2suffix = key2suffix), 
            val=dict(sample_rate = 0.5, json_filename = 'dataset_1130_truesplit.json', img_dir=img_dir, key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_1130_truesplit.json',img_dir=img_dir, key2suffix = key2suffix))

# densecl_cp = 'work_dirs/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'
model = dict(
        # backbone = dict(init_cfg=dict(
        #                 type='Pretrained', prefix='backbone.', 
        #                 checkpoint=densecl_cp), 
        #                 verbose = False, map_location='cpu'), 
        # neck = dict(type = 'FaPN3D'), 
        seg_head = dict(verbose = False),
        bbox_head = dict(verbose = False, use_vfl = True, 
                        anchor_generator = dict(verbose = False), 
                        loss_cls = dict(verbose = False), 
                            ), 
        train_cfg=dict(
            assigner=dict(topk = 4, center_within = False, verbose = False), 
            sampler=dict(num = 32, pool_size = 20, pos_fraction=0.4),
            )
        )

find_unused_parameters=True
load_from = 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_densecl/best_mAP_epoch_28.pth'
resume_from =  None # 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_vfl_densecl/latest.pth' 

# optimizer
optimizer = dict(
                type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001, 
                # _delete_ = True, type='AdamW', lr=1e-4, weight_decay=1e-4
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                # policy='poly', power=0.99, min_lr=1e-5, 
                policy='CosineAnnealing',  min_lr=1e-5,
                 by_epoch=False, warmup='linear', warmup_iters=500
                 )
runner = dict(type='EpochBasedRunner', max_epochs=16)
checkpoint_config = dict(interval=4, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=10, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                # save_best='mAP@8@0.1', 
                iou_thr=[0.2, 0.3], proposal_nums=(1, 2, 4, 8, 50))


# CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_ft.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29316 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_ft.py 3 --gpus 3 #--no-validate

# 32 epochs: mAP 0.2337
