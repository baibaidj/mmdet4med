# model1=fcn_hr18_3x448x448_40eps_lobesw_fp16_sgd_aug_grl_int16_auxsdm0.4
export PYTHONPATH=':'
gpuix=$1 #${PORT:-29500}
numfold=$2


# model_name=retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl
# weightfile=best_recall@8@0.1_epoch_24.pth


# model_name=retina_unet_repvgg_b0sd_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor
# weightfile=best_recall@8@0.1_epoch_18.pth # 2 anchors

# model_name=retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_densecl
# weightfile=best_mAP_epoch_24.pth

model_name=simmim_swint_4l16c_allct_bone_160x160x128_100eps
weightfile='latest.pth' #best_mAP_epoch_14.pth #

work_dirs=/home/dejuns/git/mmdet4med/work_dirs 
# data_rt=/home/dejuns/git/mmdet4med/data/Task113_RibFrac_Keya
# data_rt=/mnt/data4t/dejuns/ribfrac/processed/plan_rib_whole
# split=test
# setname=ky46

data_rt=/home/dejuns/git/mmdet4med/data/Task113_RibFrac_KYRe

split=test
setname=kyt30o40

gpuix=${gpuix:-0}
numfold=${numfold:-3}
foldix=${FOLDIX:-0}
# for (( foldix = 0; foldix < $numfold; foldix++ )) # $numfold
# do {
#     sleeptime=$(( 1*foldix + 0 ))
#     gpuix_exe=$gpuix #$(( gpuix + $foldix))
#     echo GPU$gpuix_exe-Foldix$foldix/$numfold-WaitStart$sleeptime
#     if [[ $foldix -gt 0 ]]; then # run in parralel but not start together
#         sleep $sleeptime
#     fi
#     CUDA_VISIBLE_DEVICES=$gpuix_exe python scripts/infer_ribfracture.py \
#     --data-rt $data_rt --repo-rt $work_dirs --pos-thresh '0.5' \
#     --model $model_name --not-ky-style --weight-file $weightfile \
#     --split $split --dataset-name $setname --gpu-ix 0 \
#     --fold-ix $foldix --num-fold $numfold --verbose --fp16
#     } &
# done
# wait

CUDA_VISIBLE_DEVICES=$gpuix python3 scripts/infer_mim.py \
    --data-rt $data_rt --repo-rt $work_dirs --pos-thresh '0.5' \
    --model $model_name --not-ky-style --weight-file $weightfile \
    --split $split --dataset-name $setname --gpu-ix 0 \
    --fold-ix $foldix --num-fold 1 --verbose --fp16 --run_pids '1666893-20201101'  #

# bash scripts/infer_mim.sh 4 1

# python tools/model_converters/publish_model.py $work_dirs/$model_name/${weightfile} $work_dirs/$model_name/publish.pth
# python tools/model_converters/publish_model.py /mnt/data2/dejuns/ribfrac/model_save/v2.7.2/fracture_det_dj.pth \
# /mnt/data2/dejuns/ribfrac/model_save/v2.7.2/fracture_det_dj_pub.pth
# python tools/model_converters/publish_model.py /mnt/d4/saved/model_save/networks_frac/fracture_det_dj.pth \
#  /mnt/d4/saved/model_save/networks_frac/fracture_det_pub.pth