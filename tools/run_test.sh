# single-gpu testing
# CUDA_VISIBLE_DEVICES=0 python tools/test.py ${CONFIG_FILE} \
#       ${CHECKPOINT_FILE} \
#       [--out ${RESULT_FILE}] \
#       [--eval ${EVAL_METRICS}] \
#       [--show]


# multi-gpu testing
# bash tools/dist_test.sh \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT_FILE} \
#     ${GPU_NUM} \
#     [--out ${RESULT_FILE}] \
#     [--eval ${EVAL_METRICS}]

task=ribfrac
# model=retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft1cls
# model=retina_unet_r34_4l16c_rf1231_160x192x128_1cls_ohem_atssdy_vfl
model=retina_unet_r34_4l16c_rf1231_160x192x128_1cls_ohem_atssdy_vfl #retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn
# model=retinanet3d_4l8c_vnet_1x_ribfrac_1cls


CUDA_VISIBLE_DEVICES=0 python tools/test_med.py configs/${task}/${model}.py \
    work_dirs/${model}/latest.pth \
    --eval mAP


# model_rt=/mnt/data4t/dejuns/ribfrac/model_save/v2.2.1
# echo mode_rt is $model_rt
# python tools/model_converters/publish_model.py $model_rt/fracture_det_dj.pth  $model_rt/fracture_det_dj


# model_rt=/mnt/data4t/dejuns/ribfrac/model_save/lung_combo/networks_nodule
# echo mode_rt is $model_rt
# python tools/model_converters/publish_model.py $model_rt/nodule_classifier_dj5cls_final.pth  $model_rt/nodule_classifier_dj5cls
# python tools/model_converters/publish_model.py $model_rt/mmseg_seg_model_best.pth  $model_rt/lunglobe_2d5cls_mmseg

# python tools/model_converters/publish_model.py $model_rt/res_bifpn_fcn2.pth  $model_rt/res_bifpn_fcn2_slim