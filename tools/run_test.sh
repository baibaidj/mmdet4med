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
model=retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft1cls


CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/$task/$model.py \
    work_dirs/$model/latest.pth \
    --out work_dirs/$model/test_results.pkl
    --eval bbox recall
