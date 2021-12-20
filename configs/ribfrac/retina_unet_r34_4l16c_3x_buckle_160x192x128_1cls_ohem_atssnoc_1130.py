_base_ = [
    '../_base_/models_med/retina_unet_r34_4l16c_atssnoc_160x192x128_vfl_buckle.py',
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
        #                 checkpoint=densecl_cp, map_location = 'cpu'), 
        #                 verbose = False), 
        # neck = dict(type = 'FaPN3D'), 
        bbox_head = dict(verbose = False, use_vfl = False, 
                        anchor_generator = dict(verbose = False), 
                        loss_cls = dict(_delete_ = True,
                                        type='CrossEntropyLoss', verbose = False,  
                                        use_sigmoid=False, class_weight = (2.0, 1.0), loss_weight=4.0), 
                        loss_bbox=dict(loss_weight=0.4), 
                            ), 
        seg_head = dict(verbose = False, 
                    loss_decode =dict(loss_weight=(1.0 * 0.2, 0.66 * 0.2))),
        train_cfg=dict(
            assigner=dict(topk = 4, center_within = False, verbose = False), 
            sampler=dict(num = 32, pool_size = 20, pos_fraction=0.4),
            )                    
        )

find_unused_parameters=True
load_from =  'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_1130/latest.pth'
resume_from =  None # 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_1130/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=1e-4, weight_decay=1e-4
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                 policy='poly', power=0.99, min_lr=1e-5, 
                # policy='CosineAnnealing',  min_lr=1e-5,
                 by_epoch=False, warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=4, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=40, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                iou_thr=[0.2, 0.3])


# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_buckle_160x192x128_1cls_ohem_atssnoc_1130.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29123 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_buckle_160x192x128_1cls_ohem_atssnoc_1130.py 3 --gpus 3 #--no-validate

# 32 epoch: 0.50@1 0.58@2 0.67@4 0.76@8 0.78@50
