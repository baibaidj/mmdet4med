_base_ = [
    '../_base_/models_med/retina_unet_r34_4l16c_atssnoc_160x192x128_tood_3cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl.py',
    # '../_base_/swa.py',
]

img_dir = 'data/Task113_RibFrac_KYRe'
key2suffix = {'img_fp': '_image.nii',  'seg_fp': '_instance.nii', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 2, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_1222.json', 
                      img_dir=img_dir, key2suffix = key2suffix, 
                    #   include_pids = ['1651036-20201010', '1620374-20200825', '1672867-20201110']
                      ), 
            val=dict(sample_rate = 0.5, json_filename = 'dataset_1222.json', img_dir=img_dir, key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_1222.json',img_dir=img_dir, key2suffix = key2suffix))

# densecl_cp = 'work_dirs/densecl_r34_4l16c_ct5k_bone_160x192x128_100eps/latest.pth'
model = dict(
        bbox_head = dict(stacked_convs=6, verbose = False, 
                        initial_loss_cls=dict(loss_weight=2.0, alpha=0.7), 
                        loss_cls = dict(loss_weight=2.0), 
                        loss_bbox=dict(loss_weight=0.7), 
                        anchor_generator=dict(octave_base_scale=2, scales_per_octave=2, verbose = False)
                            ), 
        seg_head = dict(#verbose = True, 
                        loss_decode =dict( loss_weight=(1.0 * 0.2, 0.66 * 0.2), verbose = False), 
                    ),
        train_cfg= dict(
            initial_epoch = -4, pos_weight=1.0,
            initial_assigner = dict(topk = 6, center_within = False, verbose = False), 
            assigner = dict(topk = 6, alpha = 1, beta = 4, verbose = False), 
            sampler=dict(num = 48, pool_size = 20, pos_fraction=0.4)
            ), 

        test_cfg = dict(nms_pre=300, score_thr=0.2, max_per_img=48, sw_batch_size = 6, overlap=0.4),                
    )
# custon hooks: HeadHook is defined in mmdet/core/utils/head_hook.py
custom_hooks = [dict(type="HeadHook")]

find_unused_parameters=True
load_from = 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_tood_1222/best_mAP_epoch_6.pth'
resume_from =  None # 'work_dirs/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_tood_1222/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=1e-4, weight_decay=1e-4
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
# fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
#             backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                 policy='poly', power=0.99, min_lr=1e-5, 
                # policy='CosineAnnealing',  min_lr=1e-5,
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


# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_tood_1222.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29123 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_3cls_ohem_atssnoc_tood_1222.py 3 --gpus 3 --seed 42 #--no-validate

# 32 epoch: 0.50@1 0.58@2 0.67@4 0.76@8 0.78@50

# causing nan
# data/Task113_RibFrac_KYRe/KY_B7_158_crop/1651036-20201010/1651036-20201010_instance.nii
# data/Task113_RibFrac_KYRe/KY_B8_120_crop/1620374-20200825/1620374-20200825_instance.nii
# data/Task113_RibFrac_KYRe/KY_B9_150_crop/1672867-20201110/1672867-20201110_instance.nii
# data/Task113_RibFrac_KYRe/KY_B7_158_crop/1651036-20201010/1651036-20201010_instance.nii
# data/Task113_RibFrac_KYRe/KY_B8_120_crop/1620374-20200825/1620374-20200825_instance.nii
# data/Task113_RibFrac_KYRe/KY_B9_150_crop/1672867-20201110/1672867-20201110_instance.nii