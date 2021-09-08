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
# model=retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft1cls
model=retina_unet_r34_4l8c_3x_ribfrac_160x192x128_1cls_ohem_atss_rf #retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn
# model=retinanet3d_4l8c_vnet_1x_ribfrac_1cls


CUDA_VISIBLE_DEVICES=1 python tools/test_med.py configs/$task/$model.py \
    work_dirs/$model/latest.pth \
    --eval recall
    
# CUDA_VISIBLE_DEVICES=1 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall