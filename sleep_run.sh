sleep 4h
#CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_r18_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_1anchor.py
#CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atssnoc_vfl_fapn.py

#CUDA_VISIBLE_DEVICES=1,3,5 PORT=29123 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atss_tood_upnorm_1231.py 3 --gpus 3
CUDA_VISIBLE_DEVICES=1,3,5 PORT=29010 bash ./tools/dist_train.sh configs/ribfrac/simmim_swint_4l16c_allct_bone_160x160x128_100eps_interp.py 3 --gpus 3 --no-validate
