import os.path as osp
import numpy as np
import mmcv, sys, warnings, os
from functools import reduce
from mmcv.utils import print_log
from torch.utils.data import Dataset

from ..core import cfsmat4mask_batched, metric_in_cfsmat_1by1
from .builder import DATASETS
from .pipelines import Compose
import torch, json
from pathlib import Path

import torch, pdb

# from monai.data.image_reader import NibabelReader
from .transform4med.io4med import print_tensor, random, convert_label
from .transform4med.paths import load_dataset_id
from .transform4med.load_nn import load_pickle

def decide_cid_in_fn(fn):
    fn_stem = fn.split('.')[0]
    stem_chunks = fn_stem.split('_')
    if len(stem_chunks) == 2:
        stem_chunks.append('0000')
    return '_'.join(stem_chunks[1:])

def add_series_chunk_safe(fn):
    fn_chunks = fn.split('_')
    if len(fn_chunks) == 2: return fn.replace('.nii.gz', '_0000.nii.gz')
    else: return fn


@DATASETS.register_module()
class CustomDatasetNN(Dataset):
    """Custom dataset for instance segmentation.

    An example of file structure is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 sample_rate = 1.0,
                 view_channel = None,
                 verbose = False,
                 key2suffix = {'img_fp': '.npy', 'seg_fp': '_seg.npy', 
                            'property_fp':'.pkl', 'bboxes_fp': '_boxes.pkl'},
                 keys = ('img', 'gt_instance_seg'),
                 exclude_pids = None,
                 json_filename = 'dataset.json',
                 fn_spliter = ['_', 1],
                 pos_neg_ratio = 0.5, 
                 ):

        self.key2suffix = key2suffix
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir if data_root is None else osp.join(data_root, img_dir)
        # self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        # self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
        self.view_channel = view_channel
        self.keys = keys
        self.verbose = verbose
        self.exclude_pids = exclude_pids
        self.json_filename  = json_filename
        self.fn_spliter = fn_spliter
        self.pos_neg_ratio = pos_neg_ratio

        # load annotations
        self.img_infos = self._img_list2dataset(self.img_dir, mode = self.split, key2suffix = key2suffix)
        self.img_infos = self._sample_img_data(self.img_infos, self.sample_rate)
        self.instance_cache = self.build_instance_list()
        self._set_group_flag()
        # print(self.img_infos)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.instance_cache)

    def _img_list2dataset(self, data_folder:str, mode = 'train ', 
                        key2suffix = {'img_fp': '.npy', 'seg_fp': '_seg.npy', 
                            'property_fp':'.pkl', 'bboxes_fp': '_boxes.pkl'},):
        """

        img_fp = f"{c}.npy"
            np.ndarray
        seg_fp = f"{c}_seg.npy"
            np.ndarray
        property_fp = "{c}.pkl"
            original_size_of_raw_data :	 [380 512 512]
            original_spacing :	 [1.         0.79296875 0.79296875]
            list_of_data_files :	 [PosixPath('/data/lung_algorithm/data/nnDet_data/Task020FG_RibFrac/raw_splitted/imagesTr/RibFrac343_0000.nii.gz')]
            seg_file :	 /data/lung_algorithm/data/nnDet_data/Task020FG_RibFrac/raw_splitted/labelsTr/RibFrac343.nii.gz
            itk_origin :	 (-226.603515625, -389.103515625, -822.7999877929688)
            itk_spacing :	 (0.79296875, 0.79296875, 1.0)
            itk_direction :	 (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            instances :	 {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            crop_bbox :	 [(0, 380), (0, 512), (0, 512)]
            classes :	 [-1.  0.  1.  2.  3.  4.  5.]
            size_after_cropping :	 (380, 512, 512)
            size_after_resampling :	 (304, 550, 550)
            spacing_after_resampling :	 [1.25       0.73828101 0.73828101]
            use_nonzero_mask_for_norm :	 OrderedDict([(0, False)])
        bboxes_fp = f"{c}_boxes.pkl"
            boxes :	 [[ 23 314  33 341  76  91]
                    [166 164 185 211 377 414]
                    [141 159 162 225 388 442]
                    [117 175 138 234 427 462]
                    [101 201 121 266 451 478]]
            instances :	 [1, 2, 3, 4, 5]
            labels :	 [0, 0, 0, 0, 0]

        return 
            file_list : [{'image' : img_path, 'label' : label_path}, ...]
        """
        file_list = list(), list()
        # a = [print(self.map_key(k)) for k in keys]
        js_fp = os.path.join(data_folder, self.json_filename)
        if not osp.exists(js_fp): return file_list
        with open(js_fp, 'r') as load_f:
            load_dict = json.load(load_f)
        case_fns = load_dict['training'] if mode == 'train' else load_dict['test']

        pid2pathpairs = []
        for ix, part_fn in enumerate(case_fns):
            cid = part_fn.split(os.sep)[-1]
            if ix < 3: print('Check cid', cid)
            if self.exclude_pids and (cid in self.exclude_pids): 
                print('Exclude ', cid)
                continue
            this_holder = {'cid': cid}
            for k, suffix in key2suffix.items(): 
                this_holder[k] = osp.join(data_folder, part_fn + suffix)
                if ix < 3: print(f'\t{k}', this_holder[k])
            pid2pathpairs.append(this_holder)
        pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid']) # TODO: debug
        return pathpairs_orderd

    def _sample_img_data(self, fp_list, sample_rate = 1.0, rand_seed = 42):
        if sample_rate< 1.0:
            np.random.seed(rand_seed)
            nb_total = len(fp_list)
            sample_ixs = np.arange(nb_total) if isinstance(fp_list, list) else list(fp_list.keys())
            sample_indexes = np.random.choice(sample_ixs, int(nb_total * sample_rate), 
                                            replace=False)
            sample_indexes = sorted(sample_indexes)                                
            fp_list = [fp_list[a] for a in sample_indexes]
        return fp_list
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def build_instance_list(self):
        """
        Build up cache for sampling, only cases with lesions

        Returns:
            Dict[str, List]: cache for sampling
                `case`: list with all case identifiers
                `instances`: list with tuple of (case_id, instance_id)
        """
        instance_cache = []
        print("Building Sampling Cache for Dataset")
        for cix, item in enumerate(self.img_infos):
            if cix< 2: print(item)
            instances = load_pickle(item['bboxes_fp'])["instances"] 
            if instances:
                for instance_id in instances:
                    instance_cache.append((cix, instance_id))
        return instance_cache

    # def pre_pipeline(self, results):
    #     """Prepare results dict for pipeline."""
    #     results['seg_fields'] = []

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Selects cases and instances. If instance id is -1 a random background
        patch will be sampled.

        Foreground sampling: sample uniformly from all the foreground classes
            and enforce the respective class while patch sampling.
        Background sampling: We jsut sample a random case
        

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        cix, c_ins_ix = self.instance_cache[idx]
        pos_sample_info = self.img_infos[cix]
        pos_sample_info['instance_ix'] = c_ins_ix
        
        cix4neg = random.randrange(0, len(self.img_infos))
        neg_sample_info = self.img_infos[cix4neg]
        pos_sample_info['instance_ix'] = -1
        results = [pos_sample_info, neg_sample_info]
        # self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        results = self.img_infos[idx]
        # self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        # if self.gt_seg_maps is not None:
        #     return self.gt_seg_maps
        if getattr(self, 'gt_seg_maps', None) is not None:
            return getattr(self, 'gt_seg_maps', None)

        # num_slices = self.pipeline.transforms[0].num_slice
        gt_seg_maps = {}
        for i, img_info in enumerate(self.img_infos):
            fp = Path(img_info['gt_semantic_seg'])
            subdir = str(fp.stem) #view2axis[self.view_channel]
            # img_full, af_mat = IO4Nii.read(fp, verbose=True, axis_order= None, dtype=np.uint8)
            img = self.reader.read(fp)
            img_full, meta_data = self.reader.get_data(img)
            af_mat = meta_data['original_affine']
            if i < 3: 
                print_tensor(f'\n[GT] {subdir} {self.view_channel}', img_full) #
                print('[GT] af matrix\n', af_mat)
            gt_seg_map  = np.array(img_full, dtype = np.uint8)
            # print('Eval select transform', len(result_dicts))
            # pdb.set_trace()
            gt_seg_maps.setdefault(subdir, {'gix':[], 'pix' :[], 'affine':None, 'gt': [], 'ixs': []})
            gt_seg_maps[subdir]['gix'].append(i)
            gt_seg_maps[subdir]['affine'] = af_mat


            # if i < 3: print_tensor(f'gt-convertlabel {label_mapping}' , gt_seg_map) #
            # print('eval, mask', gt_seg_map.shape, gt_seg_map.min(), gt_seg_map.max())
            gt_seg_maps[subdir]['gt'].append(gt_seg_map)
        self.gt_seg_maps = gt_seg_maps
        return gt_seg_maps

    def get_gt_map_idx(self, idx):
        load_mask_func = lambda fp: mmcv.imread(fp, flag='unchanged', backend='pillow')
        tmpf = [f for f in self.pipeline.transforms if hasattr(f, 'label_mapping')]
        label_mapping = None if len(tmpf) ==0 else tmpf[0].label_mapping

        # gt_seg_maps = []
        img_info = self.img_infos[idx]
        gt_seg_map = load_mask_func(img_info['ann']['seg_map'])
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_seg_map[gt_seg_map == 0] = 255
            gt_seg_map = gt_seg_map - 1
            gt_seg_map[gt_seg_map == 254] = 255
        if label_mapping is not None: gt_seg_map = convert_label(gt_seg_map, label_mapping)
        # print('eval, mask', gt_seg_map.shape, gt_seg_map.min(), gt_seg_map.max())
        return gt_seg_map

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset. will be called by core/evalutation/eval_hooks.py

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)


        all_acc, acc, iou, dice2d, dice3d, recall3d, precision3d = [list() for _ in range(7)]
        msg_by_pid = '\n\tpid\t\tacc\trecall\tprecision\tdice2d\tdice3d\n'
        # f24 = lambda x: '%0.4f'%f24
        # prog_bar = mmcv.ProgressBar(len(gt_seg_maps))

        verbose = True
        # reorganize the pred and gt by pids
        # find all the gix that belong to individual pids. 
        i = 0
        for pid, info in gt_seg_maps.items():
            # pid = pid.split('/')[3]
            # gt_list = info['gt']
            # pred_list = [norm_dim_4_torch(results[s], verbose= verbose) for s in info['gix']] #results[info['gix']] #
            # pred_tensor = overlap_handler_zaxis(pred_list, info['ixs'], verbose= verbose)
            pred_tensor = results[info['gix'][0]]
            gt_tensor = info['gt'][0]
            if i < 3: print_tensor(f'[Metric]{pid} pred {np.unique(pred_tensor)}', pred_tensor)
            if i < 3: print_tensor(f'[Metric]{pid} gt {np.unique(gt_tensor)}', gt_tensor)
            # store_root = '/home/dejuns/check_infer'
            # IO4Nii.write(gt_tensor, store_root, pid + '_gt', affine_matrix=info['affine'])
            # IO4Nii.write(pred_tensor, store_root, pid + '_pred', affine_matrix=info['affine'])
            # by class
            # acc_cls1 = np.sum((pred_tensor == gt_tensor) * (gt_tensor == 1))
            # pdb.set_trace()
            cfs_matrix_list = cfsmat4mask_batched(list(pred_tensor), list(gt_tensor), num_classes, self.ignore_index)
            metric2ds, metric3d = metric_in_cfsmat_1by1(cfs_matrix_list)
            gt_seg_maps[pid]['metric2ds']  = metric2ds
            gt_seg_maps[pid]['metric3d'] = metric3d
            acc_e = np.array(metric3d['acc']).round(4)
            all_acc_e = np.array(metric3d['all_acc'][0]).round(4)
            iou_e = np.array(metric3d['iou']).round(4)
            dice2d_e = np.array([a['dice'] for a in metric2ds]).mean(0).round(4)
            dice3d_e = np.array(metric3d['dice']).round(4)
            recall_e = np.array(metric3d['recall']).round(4)
            precision_e = np.array(metric3d['precision']).round(4)
            pid_short = pid ##str(pid.split('_')[1])
            msg_by_pid += f'\t{pid_short}\t{acc_e}\t{recall_e}\t{precision_e}\t{dice2d_e}\t{dice3d_e}\n'
            all_acc.append(all_acc_e)
            acc.append(acc_e); iou.append(iou_e); dice2d.append(dice2d_e);dice3d.append(dice3d_e)
            recall3d.append(recall_e); precision3d.append(precision_e)
            
            i += 1
            # prog_bar.update()
        # print(msg_by_pid)
        all_acc, acc, iou, dice2d, dice3d, recall3d, precision3d = [np.nanmean(np.array(a), axis = 0) for a in 
                                                                    [all_acc, acc, iou, dice2d, dice3d, recall3d, precision3d]]
        # print('\tcheck all acc', all_acc)

        summary_str = msg_by_pid
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc', 'Dice2d', 'Dice3d', 'Recall3d', 'Precision3d')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        remove_bg_item = lambda x: [x[i] for i in range(x.size) if class_names[i] != 'background']
        acc, iou, dice2d, dice3d = [remove_bg_item(metrics) for metrics in (acc, iou, dice2d, dice3d)]
        class_names = [a for a in class_names if a != 'background']
        
        # the resultant metrics will a list
        for i in range(len(acc)):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            dice2d_str = '{:.2f}'.format(dice2d[i] * 100)
            dice3d_str = '{:.2f}'.format(dice3d[i] * 100)
            recall3d_str = '{:.2f}'.format(recall3d[i] * 100)
            precision3d_str = '{:.2f}'.format(precision3d[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str, dice2d_str, dice3d_str, 
                                            recall3d_str, precision3d_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc', 'mDice2d', 'mDice3d', 'mRecall', 'mPrecision')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        dice2d_str = '{:.2f}'.format(np.nanmean(dice2d) * 100)
        dice3d_str = '{:.2f}'.format(np.nanmean(dice3d) * 100)
        recall3d_str = '{:.2f}'.format(np.nanmean(recall3d) * 100)
        precision3d_str = '{:.2f}'.format(np.nanmean(precision3d) * 100)
        summary_str += line_format.format('global', iou_str, acc_str, all_acc_str, 
                                        dice2d_str, dice3d_str, recall3d_str, precision3d_str)
        print_log(summary_str, logger)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc
        eval_results['mDice2d'] = np.nanmean(dice2d)
        eval_results['mDice3d'] = np.nanmean(dice3d)
        eval_results['mRecall3d'] = np.nanmean(recall3d)
        eval_results['mPrecision3d'] = np.nanmean(precision3d)
        # eval_results['mDice_obj'] = np.nanmean(dice[1:])

        return eval_results


def overlap_handler_zaxis(tensor_list, indices_list, stack_axis = 0, verbose = True):
    """
    locate overlap of zaxis for predition of a CT series 
    and take the mean for the overlap region

    tensor_list: [3dtensor_1, 3dtensor_2]

    """
    # assert len(tensor_list)
    is_torch_tensor = isinstance(tensor_list[0], torch.Tensor)
    if is_torch_tensor:
        stack_func = torch.stack
        device = tensor_list[0].device
        dim_arg = 'dim'
    else:
        stack_func = np.stack
        device = None
        dim_arg = 'axis'
    slice_pred_dict = {}
    total_slices = 0
    for tensor1, ixs in zip(tensor_list, indices_list):
        for i, z_ix in enumerate(ixs):
            pred_i = tensor1[i, ...].to(device) if device else tensor1[i, ...] # 3d tensor, 
            # if verbose: print_tensor('\tpred %d' %i, pred_i)
            slice_pred_dict.setdefault(z_ix, []).append(pred_i)
            total_slices += 1
    
    unique_z_ixs = sorted(list(slice_pred_dict))
    if verbose: print('\t total: %d, unique:' %total_slices, len(unique_z_ixs), unique_z_ixs[:3], unique_z_ixs[-3:])
    pred4d = stack_func([stack_func(slice_pred_dict[z], **{dim_arg : -1 }).sum(**{dim_arg : -1 }) / len(slice_pred_dict[z]) 
                        for z in unique_z_ixs], **{dim_arg :stack_axis})
    # pred4d = np.array(pred4d, dtype = np.uint8)
    if verbose: print_tensor('\tpred4d', pred4d)
    return pred4d


def norm_dim_4_torch(tensor, verbose = False, device = 0):
    """
    transform a 5d tensor to 4d tensor assuming the batch-size dim is 1

    """
    if verbose: print_tensor('\tNORM_DIM:', tensor)
    tensor_dim = len(tensor.shape)
    if tensor_dim == 5:
        b, c, d, h, w = tensor.shape
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.permute(0, 2, 1, 3, 4).view(b * d, c, h, w)
        else:
            tensor = tensor.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    if tensor_dim == 4:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.permute(1, 0, 2, 3)
        else:
            tensor = tensor.transpose(1, 0, 2, 3)
    return tensor

def norm_dim_4_np(tensor):
    """
    transform a 5d tensor to 4d tensor assuming the batch-size dim is 1

    """
    if len(tensor.shape) == 5:
        b, c, d, h, w = tensor.shape
        tensor = tensor.transpose(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    return tensor


denormalize = lambda x: 2.0 * x - 1.0

def logit2mask(seg_logit):
    seg_pred = seg_logit.argmax(dim=1) if seg_logit.shape[1] > 1 else seg_logit.squeeze(1) > 0.5
    seg_pred = seg_pred.int().cpu().numpy()
    return seg_pred