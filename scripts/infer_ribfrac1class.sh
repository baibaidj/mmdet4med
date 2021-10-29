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

# model_name=retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_3cls
# weightfile=latest.pth

# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_14anchor
# weightfile=best_recall@8@0.1_epoch_26.pth
model_name=retina_unet_pvt_5l16c_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_fapn
weightfile=latest.pth
# model_name=retina_vfnet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_3level
# weightfile=best_mAP_epoch_22.pth

work_dirs=/home/dejuns/git/mmdet4med/work_dirs 
# data_rt=/home/dejuns/git/mmdet4med/data/Task113_RibFrac_Keya
# data_rt=/mnt/data4t/dejuns/ribfrac/processed/plan_rib_whole
# split=test
# setname=ky46

data_rt=/home/dejuns/git/mmdet4med/data/Task113_RibFrac_Keya
# data_rt=/mnt/data4t/dejuns/ribfrac/raw_rename
# label_rt=''
# data_rt=/data/dejuns/ribfrac/validation/image
# label_rt=/data/dejuns/ribfrac/validation/gt
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
#     --data-rt $data_rt --repo-rt $work_dirs --pos-thresh '0.5' \
#     --model $model_name --not-ky-style --weight-file $weightfile \
#     --split $split --dataset-name $setname --gpu-ix 0 \
#     --fold-ix $foldix --num-fold $numfold --verbose --fp16
#     } &
# done
# wait

CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_ribfracture.py \
    --data-rt $data_rt --repo-rt $work_dirs --pos-thresh '0.5' \
    --model $model_name --not-ky-style --weight-file $weightfile \
    --split $split --dataset-name $setname --gpu-ix 0 \
    --fold-ix $foldix --num-fold 1 --verbose --fp16 --eval-final #

# bash scripts/infer_ribfrac1class.sh 1 1

# python tools/model_converters/publish_model.py $work_dirs/$model_name/best_mAP_epoch_22.pth $work_dirs/$model_name/publish.pth
# python tools/model_converters/publish_model.py /mnt/data4t/dejuns/ribfrac/model_save/v2.2.4/fracture_det_dj.pth /mnt/data4t/dejuns/ribfrac/model_save/v2.2.4/fracture_det_dj_publish.pth 
