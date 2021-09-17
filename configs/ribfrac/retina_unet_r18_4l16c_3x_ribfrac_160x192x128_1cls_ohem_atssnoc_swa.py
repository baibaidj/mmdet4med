_base_ = [
    '../ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc.py',
    '../_base_/swa.py',

]

data = dict(train=dict(sample_rate = 1.0), 
            val=dict(sample_rate = 1.0), 
            test= dict(sample_rate = 0.1))
            
model = dict(bbox_head = dict(anchor_generator = dict(verbose = False), 
                               loss_cls = dict(verbose = False), 
                                verbose = False), 
           )


swa_training = True
swa_load_from = 'work_dirs/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa/best_recall@50@0.1_epoch_6_bak.pth'
swa_resume_from = None

# swa optimizer
swa_optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# swa learning policy
swa_lr_config = dict(
    policy='cyclic',
    target_ratio=(1, 0.01),
    cyclic_times=12,
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

checkpoint_config = dict(interval=1, max_keep_ckpts = 6)
swa_checkpoint_config = dict(interval=4, filename_tmpl='swa_epoch_{}.pth', max_keep_ckpts = 4)
evaluation=dict(interval=2, start=None, 
                save_best='recall@8@0.1', rule = 'greater', 
                iou_thr=[0.1], proposal_nums=(1, 2, 4, 8, 50))
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa.py 
# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29034 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa.py 3 --gpus 3 #--no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798


#  sw_batch_size = 1, overlap = 0.25, runtime = 19.1s
#  sw_batch_size = 2, overlap = 0.25, runtime = 16.7s
#  sw_batch_size = 2, overlap = 0.33, runtime = 18.3s
#  sw_batch_size = 2, overlap = 0.5, runtime = 36.5