# dataset settings
dataset_type = 'RibFractureNN'
img_dir = 'data/Task101_RibFracture'
in_channel=  1
num_classes = 2
# 91: lung, 92: heart, 93: liver, 94: kidney
# label_mapping =  {1:1, 91:0, 92:0, 93:0, 94:0}
norm_param = {
    "mean": 330, "std": 562.5, "median": 221,
    "mn": -1024, "mx": 3071,
    "percentile_99_5": 3071,
    "percentile_00_5": -927}

keys = ('img', 'seg') #, seg='instance_seg' 
dtypes = ('float', 'int',) # , 'float', 
interp_modes = ("bilinear", "nearest")  #  , "bilinear", 'nearest'
core_key_num = 2
ext_patch_size = (192, 224, 192) # avoid artifacts such as boarder reflection
patch_size = (160, 192, 160)
rotate_angle = 20.0 
use_aux_sdm = False
train_pipeline = [
    dict(type = 'Load1CaseNN', keys = ('img', 'seg', 'property', 'bboxes'), verbose = False),  # img_meta_dict see mmseg.datasets.pipeline.transform_moani
    # dict(type = 'AddChanneld', keys= keys), 
    # dict(type = 'ConvertLabeld', keys = 'seg', label_mapping = label_mapping, value4outlier = 1), 
    dict(type = 'InstanceBasedCrop', keys=keys, patch_size= ext_patch_size, verbose = False), # cropshape
    dict(type = 'SpatialPadd', keys=keys, spatial_size= ext_patch_size, # padshape
                                mode='reflect', verbose = False),  
    dict(type = 'CastToTyped_', keys = keys,  dtype=dtypes), 
    dict(type = 'ToTensord', keys = keys),
    dict(type = 'RandFlipd_', keys = keys, spatial_axis=(0, 1), prob=0.4),
    dict(type = 'RandFlipd_', keys = keys, spatial_axis=(2, ), prob=0.4),
    # dict(type = 'RideOnLabel', keys = {'seg': ('seg', 'skeleton') }, cat_dim = 0),
    # dict(type = 'DataStatsd', keys = keys, prefix = 'Final'), 
    #  dict(type = 'SaveImaged', keys = keys, 
    #                 output_dir = '/home/dejuns/git/mmseg4med/work_dirs/AbdoVeinDataset/vnet3d_res18_4l8c_160x256x256_200eps_sw173_fp16_sgd_mimic_sup_top/debug', 
    #                 resample = False),
    dict(type='FormatShapeMonai', verbose = False, keys = keys[:core_key_num],  channels = in_channel),
    dict(type='Collect', keys=keys[:core_key_num], verbose = False, meta_keys = ('img_meta_dict', 'bboxes')),
]
test_keys = ('img', )
test_pipeline = [
        dict(type = 'Load1CaseNN', keys = test_keys), 
        dict(type='MultiScaleFlipAug3D',
            # label_mapping = label_mapping,
            # value4outlier = 1,
            target_spacings = None,  # TODO debug
            flip=False,
            flip_direction= ['diagonal'],
            transforms=[
                dict(type = 'SpacingTTAd', keys = test_keys, pixdim = None),
                dict(type = 'SpatialPadd', keys= test_keys, spatial_size= patch_size, mode='reflect', method = 'end'), 
                dict(type = 'FlipTTAd_', keys = test_keys),  
                dict(type = 'CastToTyped_', keys = 'img',  dtype= ('float', )),
                dict(type = 'ToTensord', keys = 'img'), 
                # dict(type = 'NormalizeIntensityGPUd', keys='img',
                #     subtrahend=norm_param['mean'],
                #     divisor=norm_param['std'],
                #     percentile_99_5=norm_param['percentile_99_5'],
                #     percentile_00_5=norm_param['percentile_00_5']
                #     ),
                dict(type= 'FormatShapeMonai', keys=['img'], use_datacontainer = False,
                                             channels = in_channel),
                dict(type='Collect', keys=['img'], meta_keys = ('img_meta_dict', ))
                ], 
            )
]

# draw_step = 8
total_samples = 160 #// draw_step #9842
sample_per_gpu = 2# bs2 >> 24.5 G  # 
train_sample_rate = 1.0
val_sample_rate = 0.30

data = dict(
    samples_per_gpu=sample_per_gpu,  # 16-3G
    workers_per_gpu= 2, 
    train=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = train_sample_rate, split='train',
        pipeline=train_pipeline, keys = keys, 
        # fn_spliter = ['-', 0]
        # cache_rate = 1.0, num_workers = 4
        ),
    val=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = val_sample_rate, split='test', 
        pipeline=test_pipeline,
        # fn_spliter = ['-', 0]
        ),
    test=dict(
        type=dataset_type, img_dir=img_dir, 
        sample_rate = 0.1, split='test', 
        pipeline=test_pipeline,
        # fn_spliter = ['-', 0]
        ))

gpu_aug_pipelines = [
        # # data in torch tensor, not necessarily on gpu
        dict(type = 'Rand3DElasticGPUd', keys=keys[:core_key_num],
            sigma_range=(6.2, 9.2),
            magnitude_range=(48, 81),
            spatial_size=patch_size, 
            rotate_range=[rotate_angle] * 3, #rotate_angle * np.pi / 180.0
            scale_range=(0.15, 0.15, 0.15),
            prob=1.0,
            mode=interp_modes[:core_key_num], 
            verbose=False
            ),  # 2s/mini-batch
        # dict(type = 'NormalizeIntensityGPUd', keys='img',
        #             subtrahend=norm_param['mean'],
        #             divisor=norm_param['std'],
        #             percentile_99_5=norm_param['percentile_99_5'],
        #             percentile_00_5=norm_param['percentile_00_5']
        #             ),
        dict(type = 'RandGaussianNoised_', keys='img', prob=0.4, std=0.1), 
        # dict(type = 'SaveImaged', keys = keys[:core_key_num], 
        #     output_dir = f'work_dirs/retinanet3d_4l8c_vnet_1x_ribfrac_syncbn_ft1cls/debug', 
        #     resample = False, save_batch = True, on_gpu = True),    
        ]