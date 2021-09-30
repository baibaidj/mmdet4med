# model1=fcn_hr18_3x448x448_40eps_lobesw_fp16_sgd_aug_grl_int16_auxsdm0.4
export PYTHONPATH=':'
gpuix=$1 #${PORT:-29500}
numfold=$2

# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_swa
# weightfile=latest
# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor
# weightfile=best_recall@8@0.1_epoch_32.pth

# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc
# weightfile=latest.pth

# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl
# weightfile=best_recall@8@0.1_epoch_24.pth

# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa
# weightfile=best_recall@8@0.1_epoch_8.pth

# model_name=retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_swa
# weightfile=best_recall@50@0.1_epoch_10.pth

# model_name=retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor
# weightfile=best_recall@8@0.1_epoch_18.pth # 2 anchors

# model_name=retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_adamw
# weightfile=best_recall@8@0.1_epoch_24.pth # 14 anchors

model_name=retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls
weightfile=best_mAP_epoch_12_round2.pth

repo_rt=/home/dejuns/git/mmdet4med/work_dirs 


# python tools/model_converters/publish_model.py $repo_rt/$model_name/latest.pth $repo_rt/$model_name/publish.pth
# python tools/model_converters/publish_model.py /mnt/data4t/dejuns/ribfrac/model_save/v2.2.4/fracture_det_dj.pth /mnt/data4t/dejuns/ribfrac/model_save/v2.2.4/fracture_det_dj_publish.pth 

# data_rt=/home/dejuns/git/mmdet4med/data/Task113_RibFrac_Keya
# data_rt=/mnt/data4t/dejuns/ribfrac/processed/plan_rib_whole
# label_rt=''

data_rt=/data/dejuns/ribfrac/validation/image
label_rt=/data/dejuns/ribfrac/validation/gt
split=test
setname=ky46tta

gpuix=${gpuix:-0}
numfold=${numfold:-3}
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
#     --data-rt $data_rt --label-rt $label_rt --repo-rt $repo_rt --pos-thresh '0.5' \
#     --model $model_name --not-ky-style --weight-file $weightfile \
#     --split $split --dataset-name $setname --gpu-ix 0 \
#     --fold-ix $foldix --num-fold $numfold --verbose --fp16
#     } &
# done

# # wait
CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
    --data-rt $data_rt --label-rt $label_rt --repo-rt $repo_rt --pos-thresh '0.5' \
    --model $model_name --not-ky-style --weight-file $weightfile \
    --split $split --dataset-name $setname --gpu-ix 0 \
    --fold-ix $foldix --num-fold $numfold --verbose --fp16 --eval-final #

# bash scripts/infer_ribfrac1class.sh 0 2
# python -m line_profiler 

# python tools/model_converters/publish_model.py /data/dejuns/ribfrac/model_save/v2.2.4/networks_frac/epoch_30_best.pth \
# /data/dejuns/ribfrac/model_save/v2.2.4/networks_frac/fracture_det_ft.pth