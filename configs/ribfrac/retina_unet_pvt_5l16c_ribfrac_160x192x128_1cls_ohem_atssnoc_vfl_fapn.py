_base_ = [
    '../_base_/models_med/retina_unet_pvt_5l16c_atssnoc_160x192x128_vfl_fapn.py',
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

key2suffix = {'img_fp': '_image.nii',  'seg_fp': '_instance.nii', 'roi_fp':'_ins2cls.json'}
data = dict(samples_per_gpu = 2, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, json_filename = 'dataset_1014.json', split='train', key2suffix = key2suffix), 
            val=dict(sample_rate = 1.0, json_filename = 'dataset_1014.json', split='test', key2suffix = key2suffix), 
            test= dict(sample_rate = 0.1, json_filename = 'dataset_1014.json', split='test', key2suffix = key2suffix))
            
            
model = dict(
        bbox_head = dict(
                anchor_generator = dict(verbose = False), 
                verbose = False, 
                use_vfl = True), 
            )

find_unused_parameters=True
load_from = 'work_dirs/retina_unet_pvt_5l16c_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_fapn/latest.pth'
resume_from = None # 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl/latest.pth' 

# optimizer
optimizer = dict(#type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=0.0001, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
# fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
#             backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, policy='poly', power=0.99, min_lr=1e-5, by_epoch=False,
                 warmup='linear', warmup_iters=500
                 )

runner = dict(type='EpochBasedRunner', max_epochs=32)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=30, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=4, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                # save_best='mAP@8@0.1', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))


# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/ribfrac/retina_unet_pvt_5l16c_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_fapn.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29266 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_pvt_5l16c_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_fapn.py 3 --gpus 3 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798
