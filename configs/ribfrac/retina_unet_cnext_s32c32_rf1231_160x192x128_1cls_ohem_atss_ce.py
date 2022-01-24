_base_ = [
    '../_base_/models_med/retina_unet_cnext_s32c32_atss_160x192x128_vfl_1cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/swa.py',
]

img_dir = 'data/Task113_RibFrac_KYRe'
key2suffix = {'img_fp': '_image.nii.gz',  'seg_fp': '_instance.nii.gz', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 4, workers_per_gpu= 10, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_1231.json', img_dir=img_dir, key2suffix = key2suffix), 
            val=dict(sample_rate = 0.5, json_filename = 'dataset_1231.json', img_dir=img_dir, key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_1231.json',img_dir=img_dir, key2suffix = key2suffix))

pretrain_cp = 'work_dirs/simmim_convnext_s32c32_allct_bone_160x160x128_100eps_interp/latest.pth'
model = dict(
        backbone = dict(init_cfg=dict(
                        type='Pretrained', prefix='backbone.', 
                        checkpoint=pretrain_cp, map_location = 'cpu')
                        ), 
        neck = dict(upsample_cfg=dict(use_norm = True, mode=None), 
                    init_cfg=dict(
                        type='Pretrained', prefix='neck.', 
                        checkpoint=pretrain_cp, map_location = 'cpu'), 
                    verbose = False),
        bbox_head = dict(verbose = False, #use_vfl = True, loss_cls_vfl=dict(alpha=0.75, loss_weight=8.0),
                        loss_cls = dict(_delete_ = True,
                                        type='CrossEntropyLoss', verbose = False,  
                                        use_sigmoid=False, class_weight = (2.0, 1.0), loss_weight=4.0), 
                        loss_bbox=dict(loss_weight=0.4), 
                        anchor_generator=dict(octave_base_scale=2, scales_per_octave=2, verbose = False)
                        ), 
        seg_head = dict(verbose = False, 
                        loss_decode =dict( loss_weight=(1.0 * 0.3, 0.66 * 0.3)), 
                    ),
        train_cfg= dict(
            assigner = dict(topk = 6, center_within = False,  verbose = False), 
            sampler=dict(num = 48, pool_size = 20, pos_fraction=0.4)
            ), 

        test_cfg = dict(nms_pre=300, score_thr=0.4, max_per_img=24, sw_batch_size = 6, overlap=0.4),                
    )

find_unused_parameters=True
load_from = None # 'work_dirs/retina_unet_cnext_s32c32_rf1231_160x192x128_1cls_ohem_atss_ce/latest.pth'
resume_from = None # 'work_dirs/retina_unet_cnext_s32c32_rf1231_160x192x128_1cls_ohem_atss_ce/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=1e-4, weight_decay=0.01
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                #  policy='poly', power=0.99, min_lr=1e-5, 
                policy='CosineAnnealing',  min_lr=1e-5,
                 by_epoch=False, warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=20, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                iou_thr=[0.2, 0.3])


# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_cnext_s32c32_rf1231_160x192x128_1cls_ohem_atss_ce.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29033 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_cnext_s32c32_rf1231_160x192x128_1cls_ohem_atss_ce.py 3 --gpus 3 #--no-validate

# 32 epoch: 0.50@1 0.58@2 0.67@4 0.76@8 0.78@50
