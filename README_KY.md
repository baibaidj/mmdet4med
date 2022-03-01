
# 代码结构

~~~
git/mmdet4med: 
    ├── configs # 配置py文件，可继承
    ├── data # 存放训练数据软连接
    ├── dcn
    ├── demo # ipynb 拆解一些模块的notebook，也可以作为测试
    ├── docker
    ├── docs # 官方文档，中英文都有。
    ├── LICENSE
    ├── mmdet # 模块化训练和推理核心代码
    ├── monai # 引自monai，借用了其中的医学图像加载模块以及三位图像前处理。
    ├── README_KY.md # 
    ├── README.md
    ├── README_zh-CN.md
    ├── requirements # 依赖
    ├── requirements.txt
    ├── scripts # 自建代码，包括批量推理和测试。
    ├── tests
    ├── tools # 启动训练和测试的python脚本。
    └── work_dirs # 存放训练结果，按配置文件名
~~~

理解MMCV框架的核心模块参见 https://mmcv.readthedocs.io/en/latest/
~~~
Config
    Inherit from base config without overlapped keys
    Inherit from base config with overlapped keys
    Inherit from base config with ignored fields
    Inherit from multiple base configs (the base configs should not contain the same keys)
    Reference variables from base
    Add deprecation information in configs
Registry
    What is registry
    A Simple Example
    Customize Build Function
    Hierarchy Registry
Runner
    EpochBasedRunner
    IterBasedRunner
    A Simple Example
File IO
    Load and dump data
    Load a text file as a list or dict
    Load and dump checkpoints
~~~

# 训练

config:
~~~
./configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atss_vfl_upnorm_1231.py
~~~

文件结尾包含了启动训练的命令

单卡训练命令
~~~
CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atss_vfl_upnorm_1231.py
~~~ 

多卡训练命令 
~~~
CUDA_VISIBLE_DEVICES=1,3,5 PORT=29024 bash ./tools/dist_train.sh configs/ribfrac/retina_unet_r34_4l16c_3x_ribfrac_160x192x128_1cls_ohem_atss_vfl_upnorm_1231.py 3
~~~


训练脚本主程序
~~~
# ./tools/train.py
    cfg = Config.fromfile(args.config)
    ...
    # torch.autograd.set_detect_anomaly(True) # 需要debug nan 或者 inf时用。debug时最好单卡。 
    ...

    model = build_detector( # 初始化模型
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights() # 初始化权重
    ...
    datasets = [build_dataset(cfg.data.train)] # # 初始化训练数据集
    ...
    train_detector_func = train_detector_swa if cfg.get('swa_training', False) else train_detector
    train_detector_func(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
~~~


~~~
# .mmdet/apis/train.py

    # 抽象出来的迭代训练器，集成了数据集、模型、优化器、记录和训练流程等。
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    
    # 根据单卡或多卡来初始化 dataloader，继承自pytorch，增加了一些功能。
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # put model on gpus，优先使用DDP (Distributed Data Parallel)
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner， 构建优化器以及 训练迭代器
    optimizer = build_optimizer(model, cfg.optimizer) #SGD, AdamW

    # EpochBasedRunner 详细流程参见PPT。
    runner : EpochBasedRunner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # fp16 setting # 混合精度训练 以 钩子的方式加入 训练迭代器
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks，控制训练流程的一些配置都以钩子的方式传入。
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    # register eval hooks # 训练期间的验证也是以钩子方式传入的。
    if validate:
        ...
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHookMed if distributed else EvalHookMed # 这里做了适应性的改动。
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    ...
    # resume_from是继续训练。load_from是利用训练过的模型做微调；
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # 一切准备就绪，正式开始训练流程
    runner.run(data_loaders, cfg.workflow)  

~~~


# 训练后推理

批量推理看指标
~~~
task=ribfrac
model=retina_unet_r34_4l16c_rf1231_160x192x128_1cls_ohem_atssdy_vfl
CUDA_VISIBLE_DEVICES=0 python tools/test_med.py configs/${task}/${model}.py \
    work_dirs/${model}/latest.pth \
    --eval mAP
~~~

独立推理，可用于部署

~~~ 
bash scripts/infer_ribfrac1class.sh $gpuix $numfold
~~~

输出结果
~~~

 ├── visual_kyt30o40_test_cutoff0.5_nii
 │   ├── 1119674_20180511_fp16_det_roi_infos.json # 检测头预测的病灶信息
 │   ├── 1119674_20180511_fp16_roi_det_mask.nii.gz # 检测头预测的病灶mask，用做可视化
 │   ├── 1119674_20180511_fp16_roi_seg_mask.nii.gz # 分割头预测的病灶mas，用做可视化
 │   ├── 1119674_20180511_fp16_seg_roi_infos.json # 分割头预测的病灶信息 
 │   ├── aug_inference_0@2.csv # 每个case的检测指标，包括TP、FP和FN的个数，检测的recall和precision。分两批预测中的第一批。
 │   ├── aug_inference_1@2.csv # 每个case的检测指标，包括TP、FP和FN的个数，检测的recall和precision。分两批预测中的第二批。
 │   ├── aug_inference_all.csv # 所有case的检测指标
 │   ├── fp16_det_roi_infos_FROC.pdf # 测试集上检测头的FROC曲线
 │   ├── fp16_seg_roi_infos_FROC.pdf # 测试集上分割头的FROC曲线
 │   ....
 │   ├── RibFrac98_fp16_det_roi_infos.json
 │   ├── RibFrac98_fp16_roi_det_mask.nii.gz
 │   ├── RibFrac98_fp16_roi_seg_mask.nii.gz
 │   └── RibFrac98_fp16_seg_roi_infos.json
 └── visual_kyt30o40_train_cutoff0.5_nii
     ├── 1648270-20201005_fp16_det_roi_infos.json
     ├── 1648270-20201005_fp16_roi_det_mask.nii.gz
     ├── 1648270-20201005_fp16_roi_seg_mask.nii.gz
     ├── 1648270-20201005_fp16_seg_roi_infos.json
     └── aug_inference_0@1.csv
~~~


测试阶段配置参数
~~~
    test=dict(
        type='RibFractureDet',
        img_dir='data/Task113_RibFrac_KYRe',
        sample_rate=0.1,
        split='test',
        pipeline=[
            dict(type='Load1CaseDet',
                keys=('img', 'roi'),
                label_map=dict({1: 0, 2: 0, 3: 0, 4: 0, 7: 0})),
            dict(type='MultiScaleFlipAug3D',
                target_spacings=None,  # 指定推理时对数据进行的resize次数。[None, (1.2, 1.2, 1.2), ]
                flip=False, # 推理时是否对图像做翻转
                flip_direction=['diagonal'], # 翻转的方式，可能取值 ['diagonal', 'headfeet', 'all3axis']
                transforms=[
                    dict(type='AddChanneld', keys=('img', )),
                    dict(type='SpacingTTAd', # 根据target_spacings对原图进行resize
                        keys=('img', ),
                        pixdim=None,
                        mode=('bilinear', )),
                    dict(type='SpatialPadd_',
                        keys=('img', ),
                        spatial_size=(160, 192, 128),
                        mode='reflect',
                        method='end'),
                    dict(type='FlipTTAd_', keys=('img', )),
                    dict(type='CastToTyped_', keys=('img', ),
                        dtype=('float', )),
                    dict(type='ToTensord', keys=('img', )),
                    dict(type='NormalizeIntensityGPUd',
                        keys='img',
                        subtrahend=330,
                        divisor=562.5,
                        percentile_99_5=3071,
                        percentile_00_5=-927),
                    dict(type='FormatShapeMonai',
                        keys=('img', ),
                        use_datacontainer=False,
                        channels=1),
                    dict(type='Collect',
                        keys=('img', ),
                        meta_keys=('img_meta_dict', ))
                ])
        ],
...
test_cfg=dict(
         nms_pre=300, # 推理每个patch后通过最大化抑制保留的检测框
         min_bbox_size=1, # 检测框的体积必须大于这个值，否则排除。
         score_thr=0.1, # 检测框的分类概率必须大于这个值，否则排除。
         nms=dict(type='nms', iou_threshold=0.1), # NMS后处理中的重叠阈值
         max_per_img=48, # 一套CT图最多留多少个检测框。每个patch保留的检测框减半。
         mode='slide', # 预测模型用滑窗
         roi_size=(160, 192, 128), # 预测时的patch_size
         sw_batch_size=6, # 一套CT切成多个patch后，每次推理放几个patch
         overlap=0.4, # 滑窗推理时，相邻两个patch的重叠程度，介于[0~1]之间，越大推理越慢。
         blend_mode='gaussian',
         )
~~~


发布训练权重，仅保留模型权重，可节省接近一半空间。
~~~
python tools/model_converters/publish_model.py $work_dirs/$model_name/${weightfile} $work_dirs/$model_name/publish.pth
~~~



# 