# model1=fcn_hr18_3x448x448_40eps_lobesw_fp16_sgd_aug_grl_int16_auxsdm0.4
export PYTHONPATH=':'
gpuix=$1 #${PORT:-29500}
numfold=$2

model_name=retinanet3d_4l8c_vnet_3x_ribfrac_1cls_ohem_atss_rf
repo_rt=/home/dejuns/git/mmdet4med/work_dirs
taskname=ribfrac_det

# python tools/model_converters/publish_model.py $repo_rt/$model_name/latest.pth $repo_rt/$model_name/publish.pth

data_rt=/data/lung_algorithm/data/DetFrac/processed/plan_rib_crop
split=test
setname=ky56

gpuix=${gpuix:-0}
numfold=${numfold:-20}
foldix=${FOLDIX:-0}
# for (( foldix = 0; foldix < $numfold; foldix++ ))
# do {
#     sleeptime=$(( 1*foldix + 0 ))
#     gpuix=$foldix
#     echo GPU$gpuix-Foldix$foldix/$numfold-WaitStart$sleeptime
#     if [[ $foldix -gt 0 ]]; then # run in parralel but not start together
#         sleep $sleeptime
#     fi
#     echo GPU$gpuix #kernprof -l -v
#     CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
#     --data-rt $data_rt --repo-rt $repo_rt --pos-thresh '0.5' \
#     --model $model_name --not-ky-style --weight-file 'publish-16bdb70a.pth' \
#     --split $split --dataset-name $setname --gpu-ix 0 --seed 0 \
#     --fold-ix $foldix --num-fold $numfold --verbose 
#     } &
# done

CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
    --data-rt $data_rt --repo-rt $repo_rt --pos-thresh '0.5' \
    --model $model_name --not-ky-style --weight-file 'publish-16bdb70a.pth' \
    --split $split --dataset-name $setname --gpu-ix 0 --seed 0 \
    --fold-ix $foldix --num-fold $numfold #--eval-final #--verbose 

# bash scripts/infer_ribfrac1class.sh 5 5
# python -m line_profiler 