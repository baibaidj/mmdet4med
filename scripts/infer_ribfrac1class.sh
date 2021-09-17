# model1=fcn_hr18_3x448x448_40eps_lobesw_fp16_sgd_aug_grl_int16_auxsdm0.4
export PYTHONPATH=':'
gpuix=$1 #${PORT:-29500}
numfold=$2

model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa
weightfile=best_recall@50@0.1_epoch_6_bak.pth

# model_name=retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa
# weightfile=best_recall@50@0.1_epoch_10.pth

repo_rt=/home/dejuns/git/mmdet4med/work_dirs #best_recall@50@0.1_epoch_10.pth #latest.pth #'publish-4ff54096.pth' 
taskname=ribfrac_det


# python tools/model_converters/publish_model.py $repo_rt/$model_name/latest.pth $repo_rt/$model_name/publish.pth

data_rt=/data/lung_algorithm/data/DetFrac/processed_new/plan_rib_crop
split=test
setname=ky46

gpuix=${gpuix:-0}
numfold=${numfold:-20}
foldix=${FOLDIX:-0}
# for (( foldix = 0; foldix < $numfold; foldix++ )) # $numfold
# do {
#     sleeptime=$(( 1*foldix + 0 ))
#     # gpuix=$foldix
#     echo GPU$gpuix-Foldix$foldix/$numfold-WaitStart$sleeptime
#     if [[ $foldix -gt 0 ]]; then # run in parralel but not start together
#         sleep $sleeptime
#     fi
#     echo GPU$gpuix #kernprof -l -v
#     CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
#     --data-rt $data_rt --repo-rt $repo_rt --pos-thresh '0.5' \
#     --model $model_name --not-ky-style --weight-file $weightfile \
#     --split $split --dataset-name $setname --gpu-ix 0 \
#     --fold-ix $foldix --num-fold $numfold --verbose #--fp16
#     } &
# done

CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
    --data-rt $data_rt --repo-rt $repo_rt --pos-thresh '0.5' \
    --model $model_name --not-ky-style --weight-file $weightfile \
    --split $split --dataset-name $setname --gpu-ix 0 \
    --fold-ix $foldix --num-fold $numfold  --verbose --eval-final #

# bash scripts/infer_ribfrac1class.sh 3 3
# python -m line_profiler 