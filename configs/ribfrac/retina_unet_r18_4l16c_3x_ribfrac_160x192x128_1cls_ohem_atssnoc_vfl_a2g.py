_base_ = [
    '../_base_/models/retina_unet_r18_4l16c_atssnoc_160x192x128_vfl_1anchor.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py', 
    # '../_base_/swa.py',
]

key2suffix = {'img_fp': '_image.nii.gz',  'seg_fp': '_instance.nii.gz', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 3, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_0928.json', split='train', key2suffix = key2suffix), 
            val=dict(sample_rate = 1.0, json_filename = 'dataset_0928.json', split='test', key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_0928.json', split='test', key2suffix = key2suffix))
            
            
model = dict(backbone = dict(verbose = False ), 
            seg_head = dict(verbose = False),
            bbox_head = dict(verbose = False, use_vfl = True, 
                            anchor_generator = dict(verbose = False), 
                            loss_cls = dict(verbose = False), 
                             ))

find_unused_parameters=True
load_from = 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_ft/best_recall@8@0.1_epoch_20.pth'
resume_from = None #'work_dirs/retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_rf/latest.pth' 

# optimizer
optimizer = dict(#type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=0.0002, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=5e-5, by_epoch=False,
                 warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=30, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=4, 
                save_best='recall@8@0.1', rule = 'greater', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))

# swa_training = False
# swa_load_from = 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_swa/swa_runavg_model_2.pth'
# swa_resume_from = None # 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_swa/swa_runavg_model_2.pth'
# # swa optimizer
# swa_optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
# # swa learning policy
# swa_lr_config = dict(
#     policy='cyclic',
#     target_ratio=(1, 0.01),
#     cyclic_times=12,
#     step_ratio_up=0.0)
# swa_runner = dict(type='EpochBasedRunner', max_epochs=16)  
# swa_interval = 2
# # swa_optimizer_config
# swa_optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 8, norm_type = 2)) # 31G
# swa_checkpoint_config = dict(interval=4, filename_tmpl='swa_epoch_{}.pth', max_keep_ckpts = 4)
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_a2g.py 
# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29034 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_swa.py 3 --gpus 3 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798
# recall@1@0.1: 0.1538, recall@2@0.1: 0.2747, recall@4@0.1: 0.4341, recall@8@0.1: 0.6319, recall@50@0.1: 0.7747
# 

#  sw_batch_size = 1, overlap = 0.25, runtime = 19.1s
#  sw_batch_size = 2, overlap = 0.25, runtime = 16.7s
#  sw_batch_size = 2, overlap = 0.33, runtime = 18.3s
#   sw_batch_size = 2, overlap = 0.5, runtime = 36.5