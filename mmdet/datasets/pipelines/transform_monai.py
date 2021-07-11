
from monai.transforms import (
    LoadImaged,
    Spacingd,
    AddChanneld,
    CastToTyped_,
    SqueezeDimd,
    SpatialPadd,
    RandGaussianNoised,
    RandRotate90d,
    ToTensord,
    Rand3DElasticd,

    # self definition
    NormalizeIntensityGPUd,
    RandCropByLabelBBoxRegiond,
    # RandCropByPosNegMultiLabeld,
    RandFlipd_,
    RandGaussianNoised_,
    SelectChanneld,

    # add by dejuns
    ConvertLabeld,
    RideOnLabel,
    DataStatsd,
    CenterSpatialCropd_,
    SaveImaged, 
    Rand3DElasticGPUd, 
    ConvertLabelRadiald, 
    SpacingTTAd, 
    FlipTTAd_
)

from ..builder import PIPELINES

preg = PIPELINES.register_module()

LoadImaged = preg(LoadImaged)
Spacingd = preg(Spacingd)
AddChanneld = preg(AddChanneld)
SqueezeDimd = preg(SqueezeDimd)
CastToTyped_ = preg(CastToTyped_)
SpatialPadd = preg(SpatialPadd)
RandGaussianNoised = preg(RandGaussianNoised) 
ToTensord = preg(ToTensord)
Rand3DElasticd = preg(Rand3DElasticd)

NormalizeIntensityGPUd = preg(NormalizeIntensityGPUd)
RandCropByLabelBBoxRegiond = preg(RandCropByLabelBBoxRegiond)
CenterSpatialCropd_ = preg(CenterSpatialCropd_)
Rand3DElasticGPUd = preg(Rand3DElasticGPUd)

RandGaussianNoised_ = preg(RandGaussianNoised_)
RandFlipd_ = preg(RandFlipd_)

SelectChanneld = preg(SelectChanneld)
ConvertLabeld = preg(ConvertLabeld)
ConvertLabelRadiald = preg(ConvertLabelRadiald)
RideOnLabel = preg(RideOnLabel)
DataStatsd = preg(DataStatsd)
SpacingTTAd = preg(SpacingTTAd)
FlipTTAd_ = preg(FlipTTAd_)
SaveImaged = preg(SaveImaged)





# dict_keys(['image', 'label', 'dtmap', 'skeleton', 'seg_fields', 'image_meta_dict', 'label_meta_dict', 'dtmap_meta_dict', 'skeleton_meta_dict'])
# image_meta_dict example: 
# {'sizeof_hdr': array(348, dtype=int32), 
# 'extents': array(0, dtype=int32), 
# 'session_error': array(0, dtype=int16), 
# 'dim_info': array(0, dtype=uint8), 
# 'dim': array([  3, 441, 441, 234,   1,   1,   1,   1], dtype=int16), 
# 'in$ent_p1': array(0., dtype=float32), 
# 'intent_p2': array(0., dtype=float32), 
# 'intent_p3': array(0., dtype=float32), 
# 'intent_code': array(0, dtype=int16),
# 'datatype': array(4, dtype=int16), 
# 'bitpix': array(16, dtype=int16), 
# '$lice_start': array(0, dtype=int16), 
# 'pixdim': array([1.   , 0.725, 0.725, 1.   , 0.   , 0.   , 0.   , 0.   ], dtype=float32), 
# 'vox_offset': array(0., dtype=float32), 
# 'scl_slope': array(nan, dtype=float32), 
# 'scl_inter': array(nan, dtype=float32), 
# 'slice_end': array(0, dtype=int16), 
# 'slice_code': array(0, dtype=uint8), 
# 'xyzt_$nits': array(2, dtype=uint8), 
# 'cal_max': array(0., dtype=float32), 
# 'cal_min': array(0., dtype=float32), 
# 'slice_duration': array(0., dtype=float32), 
# 'toffset': array(0., dtype=float32), 'glmax': array(0, dtype=int32), 
# 'glm$n': array(0, dtype=int32), 'qform_code': array(1, dtype=int16), 
# 'sform_code': array(0, dtype=int16), 
# 'quatern_b': array(0., dtype=float32), 
# 'quatern_c': array(0., dtype=float32), 
# 'quatern_d': array(1., dtype=float32), 
# 'qo$fset_x': array(-0., dtype=float32), 
# 'qoffset_y': array(-0., dtype=float32), 
# 'qoffset_z': array(0., dtype=float32), 
# 'srow_x': array([0., 0., 0., 0.], dtype=float32), 
# 'srow_y': array([0., 0., 0., 0.], dtype=float32), 
# 'srow_z': array([0., 0., 0., 0.], dtype=float32), 
# 'affine': array(
#       [[-0.72500002,  0.        ,  0.        , -0.        ],
#        [ 0.        , -0.72500002,  0.        , -0.        ],
#        [ 0.        ,  0.        ,  1.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
# 
# 'original_affine': array(
#       [[-0.72500002,  0.        ,  0.        , -0.        ],
#        [ 0.        , -0.72500002,  0.        , -0.        ],
#        [ 0.        ,  0.        ,  1.        ,  0.        ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]]), 
# 'as_closest_canonical': False, 
# 'spatial_shape': array([441, 441, 234], dtype=int16), 
# 'filename_or_obj': 'data/Task011_abdomen_anatomy/imagesTr/case_00489_0000.nii.gz'}