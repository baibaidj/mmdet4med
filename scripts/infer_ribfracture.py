
import pandas as pd
import os, sys, argparse

import SimpleITK, json
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from evaluate_det_result import roi_info_dets, roi_info_mask, evaluate1case, evaluate_pred_against_label
from mmdet.apis.inference_med import *
from mmdet.datasets.transform4med.io4med import *
from scripts.froc_ import calculate_FROC_by_center
import cc3d, ipdb
from mmcv import Timer
from mmcv.runner import wrap_fp16_model
# CLASSES = ('bg', 'hv', 'pv')

def _remove_low_probs(pred, prob_thresh = 0.5):
    pred =  np.where(pred > prob_thresh, pred, 0)
    return pred


def locate_spine_region(image_3d, bone_area = None, axis_this = 2, bone_thresh = 300, 
                        spine_div = 4, spine_thresh = 0.1):

    if bone_area is None: bone_area = image_3d > bone_thresh
    # proj_raw_2d = np.mean(image_3d, axis = axis_this)

    proj_bone_2d = np.sum(bone_area, axis = axis_this)
    image_bone_2d = ndimage.median_filter(proj_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // spine_div)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area, max_region = 0, None
    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    spine_largest_convex = np.zeros_like(image_spine)
    if max_region is not None:
        spine_largest_convex[
            max_region.bbox[0]:max_region.bbox[2],
            max_region.bbox[1]:max_region.bbox[3]
        ] = max_region.convex_image > 0
    return spine_largest_convex


def _remove_spine_fp(pred, image_spine_2d = None, image_3d = None):
    assert not (image_spine_2d is None and image_3d is None)
    if image_spine_2d is None: image_spine_2d = locate_spine_region(image_3d)
    return np.where(image_spine_2d[..., None], 0, pred)


def _remove_small_objects(pred, size_thresh = 100):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def post_process_ribfrac(pred_4d, image_spine_2d = None, image = None, fg_channel = 1, 
                        prob_thresh = 0.5, bone_thresh = 300, size_thresh = 100, 
                        ):
    """
    pred_4d: CHWD, C = fg+bg
    image: HWD, 
    prob_thresh: 
    bone_thresh:
    size_thresh: 

    """
    pred_prob = pred_4d[fg_channel]
    
    # remove connected regions with low confidence
    pred_prob = _remove_low_probs(pred_prob, prob_thresh)
    # print_tensor(f'@@remove low prob thresh {prob_thresh}', pred)
    # remove spine false positives
    pred_prob = _remove_spine_fp(pred_prob, image_spine_2d = image_spine_2d, image_3d = image)
    # print_tensor(f'@@remove spine', pred)
    # remove small connected regions
    pred_prob = _remove_small_objects(pred_prob, size_thresh)
    # print_tensor(f'@@remove small objects', pred)
    return pred_prob


def _make_submission_files(pred_prob_3d, image_id):
    pred_label = cc3d.connected_components(pred_prob_3d > 0).astype(np.uint8)
    pred_regions = regionprops(pred_label, pred_prob_3d)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })
    print(f'caseid {image_id}\n', pred_info)

    return pred_label, pred_info

def main(cfg, 
        nii_save_dir = '',  
        pid2niifp_map = dict(),
        is_test = False,
        fold_ix_str = '0@1', 
        ):

    # model configuration prepare
    git_rt = Path(cfg.repo_rt)
    # model_name = 'fcn_hr18_449x449_40k_pancreas_neg100_dl'; process_func = process_1_case_2d
    model_store_dir = git_rt/cfg.model_name
    config_file = model_store_dir/str('%s.py' %cfg.model_name)
    # pdb.set_trace()
    checkpoint_file = model_store_dir/cfg.weight_file
    assert os.path.exists(checkpoint_file), '%s not exists' % str(checkpoint_file)
    # cfg_model = Config.fromfile(config_file)

    print(config_file)
    print(checkpoint_file)
    # build the model from a config file and a checkpoint file
    # with Timer(print_tmpl='Loading model %s takes {:.3f} seconds' %(cfg.model_name)):
    model = init_detector(str(config_file), str(checkpoint_file), device='cuda:%d' %cfg.gpu_ix)
    label_map = model.cfg.get('label_map', None)
    if cfg.fp16: 
        wrap_fp16_model(model)
        print('\nFP16 inference')

    # Inferer.seg_prob2save = MethodType(seg_prob2save_det, Inferer)
    # print('\nCases to infer : %d \n' %len(pid2niifp_map))
    pcount = 0
    detroi_by_case = []
    segroi_by_case = []
    gtroi_by_case = []
    result_by_case = []
    # infer by model
    for cid, cid_infos in pid2niifp_map.items(): #[1:2]
        img_nii_fp = cid_infos['image']
        # if cid != '1112514_20180429': continue  #'1214845_20181014'  #3837016_image.nii.gz
        print(f'\n[INFER] {pcount} {img_nii_fp}')
        if is_test: continue
        # store_dir
        if cfg.fp16: cid_infos['cid'] = f'{cid}_fp16'
        cid_true = cid_infos['cid']
        det_result_fp = osp.join(nii_save_dir, f'{cid_true}_roi_det_mask.nii.gz')
        if osp.exists(det_result_fp): 
            print(f'{det_result_fp} exist, so skip inference')
            continue

        img_3d, affine_matrix_i = IO4Nii.read(img_nii_fp, axis_order = None, 
                                                verbose = False, dtype=np.int16)
        cid_infos['image_3d'] = img_3d
        cid_infos['affine'] = affine_matrix_i
        timer = Timer() 
        # NOTE: make the target spacings adpative to the affine_matrix 
        origin_spacing = [abs(affine_matrix_i[a, a]) for a in range(3)]
        target_spacings = find_target_spacing(origin_spacing, use_double_spacing=True)
        print(f'origin spacing {origin_spacing} target spacint {target_spacings}')

        det_results, seg_results = inference_detector4med(model, img_3d, 
                                                        affine = affine_matrix_i,
                                                        rescale = True,
                                                        need_probs = True, 
                                                        target_spacings=target_spacings)
        seg_results = seg_results.numpy()                                            
        torch.cuda.empty_cache()
        if cfg.verbose: print_tensor('\tseg logits', seg_results)
        duration  = timer.since_start()
        
        (case_result, det_roi_infos, seg_roi_infos, gt_det_infos) = \
                evaluate_store_prediction_1case(det_results, seg_results,
                                                cid_infos, nii_save_dir, 
                                                seg_prob_thresh=cfg.pos_thresh,
                                                label_map = label_map)
        case_result['infertime'] = duration
        result_by_case.append(case_result)
        detroi_by_case.append(det_roi_infos)
        segroi_by_case.append(seg_roi_infos)
        gtroi_by_case.append(gt_det_infos)
        pcount += 1

    per_case_fp = osp.join(nii_save_dir, f'aug_inference_{fold_ix_str}.csv')
    result_tb = pd.DataFrame(result_by_case)
    result_tb.to_csv(per_case_fp, index = False)
    print(result_tb.describe())
    return per_case_fp


def find_target_spacing(spacing_xyz, 
                        target_spacing_scheme = {0: (0.725, 0.825), 1 : (0.725, 0.825), 2 : (0.94, 1.26)}, 
                        use_double_spacing = False):

    def is_value_within(v, range_):
        assert len(range_) == 2, f'range can only contain two elements but got {range_}'
        # if not isinstance(range_, (tuple, list)):
        #     range_ = [range_] * 2
        is_within = (v >= range_[0]) and (v <= range_[1])
        # print('[IS_Within]', v, range_, is_within)
        return is_within

    def find_cloest_boundary(v, range_):
        assert len(range_) == 2, f'range can only contain two elements but got {range_}'
        # if not isinstance(range_, (tuple, list)):
        #     range_ = [range_] * 2
        if is_value_within(v, range_):
            return v
        else: return sum(range_)/2

    if target_spacing_scheme is None:  
        target_spacing_inner = [[abs(s), abs(s)] for i, s in enumerate(spacing_xyz)]
    else: target_spacing_inner = target_spacing_scheme
    is_spacings_within = [is_value_within(abs(spacing_xyz[i]), target_spacing_inner[i]) 
                                                             for i in range(3)]

    # is_new_exist = os.path.exists(new_img_path)
    if all(is_spacings_within): 
        target_spacings = [None]
    else:
        target_spacing = tuple([find_cloest_boundary(abs(spacing_xyz[i]), 
                                    target_spacing_inner[i]) for i in range(3)])
        target_spacings = [target_spacing] + ([None] if use_double_spacing else [])
    return target_spacings




def evaluate_store_prediction_1case(det_results, seg_results, cid_infos, nii_save_dir, 
                                    seg_prob_thresh = 0.5, label_map = None, 
                                    gt_valid_class = (1, 2, 3, 4, 7)):
    """
    Args:
        det_results: list of detection results of 1 image
        seg_results: 5d tensor
    """

    roi_fp, seg_gt_fp = cid_infos['roi'], cid_infos['label']
    cid, img_3d, affine_mat = cid_infos['cid'], cid_infos['image_3d'], cid_infos['affine']
    seg_channel = seg_results.shape[1]

    # 1. prepare the detection target and segmentation target from either json file and mask labels
    if isinstance(roi_fp, (str, Path)) and osp.exists(roi_fp):
        gt_det_infos = load2json(cid_infos['roi'])
    else: gt_det_infos = []

    if isinstance(seg_gt_fp, (str, Path)) and osp.exists(seg_gt_fp):
        gt_roi_mask, af_mat = IO4Nii.read(seg_gt_fp, axis_order= None, verbose = False, dtype=np.uint8)
        print_tensor('\tGTmask', gt_roi_mask)
        if len(gt_det_infos) == 0 :  gt_det_infos = roi_info_mask(gt_roi_mask)
        if label_map is not None: 
            gt_roi_mask = convert_label(gt_roi_mask, label_mapping = label_map, value4outlier=1)
    else: gt_roi_mask = None

    if isinstance(gt_valid_class, (tuple, list)): 
        gt_det_infos = [roi for roi in gt_det_infos if roi['class'] in gt_valid_class]

    # 2. compute metrics for detection head first 
    det_roi_infos = roi_info_dets(det_results[0])
    # recall, precision, fp, fn, tp
    metric_keys = ('recall', 'precision', 'fp', 'fn', 'tp')
    det_roi_metrics = evaluate1case(det_roi_infos, gt_det_infos)

    det_result = {f'det_{k}': det_roi_metrics[i]  for i, k in  enumerate(metric_keys)}
    # pdb.set_trace()
    det_roi_mask = np.zeros_like(img_3d, dtype = np.uint16)
    for roi in det_roi_infos:
        # slicer = tuple([slice(roi['bbox'][i], roi['bbox'][i + 3])  for i in range(3)])
        bbox2slicer = lambda b: tuple([slice(round(b[i]), round(b[i + 3]))  for i in range(3)])
        slicer = bbox2slicer(roi['bbox'])
        cls_base = roi['class'] * 100
        det_roi_mask[slicer] = int(round(roi['prob'] * 100) + cls_base)
    IO4Nii.write(det_roi_mask, nii_save_dir, f'{cid}_roi_det_mask', affine_mat)
    save2json(det_roi_infos, nii_save_dir, f'{cid}_det_roi_infos.json', indent=None, sort_keys=False)
    # 3. compute metrics for segmentation head
    if seg_channel > 2:
        seg_roi_mask = np.argmax(seg_results[0], axis = 0).astype(np.uint8)
        seg_roi_infos = roi_info_mask(seg_roi_mask,  is_prob = False)
    else:
        seg_roi_prob = post_process_ribfrac(seg_results[0], image_spine_2d=locate_spine_region(img_3d), 
                                        prob_thresh=seg_prob_thresh)
        seg_roi_mask = np.array(seg_roi_prob > 0, dtype = np.uint8)
        seg_roi_infos = roi_info_mask(seg_roi_prob,  is_prob = True)
    # pdb.set_trace()
    print_tensor('\timage', img_3d)
    print_tensor('\tpred', seg_results)
    seg_roi_metrics = evaluate1case(seg_roi_infos, gt_det_infos)
    seg_result = {f'seg_{k}': seg_roi_metrics[i]  for i, k in  enumerate(metric_keys)}
    IO4Nii.write(seg_roi_mask, nii_save_dir, f'{cid}_roi_seg_mask', affine_mat)
    save2json(seg_roi_infos, nii_save_dir, f'{cid}_seg_roi_infos.json', indent=None, sort_keys=False)

    dice3d = evaluate_pred_against_label(seg_roi_mask, gt_roi_mask, ('dice', ), num_classes = seg_channel)['dice']
    dice_by_cls = {f'seg_dice{i}': d for i, d in enumerate(dice3d)}
    # 'seg_dice': float(dice3d)
    img_dim = {f'img_dim{i}': img_3d.shape[i] for i in range(3)}
    space_dim = {f'spacing_dim{i}': affine_mat[i, i] for i in range(3)}
    case_result = {'cid' : cid, **img_dim, **space_dim,
                    **det_result, **seg_result, **dice_by_cls}
    return case_result, det_roi_infos, seg_roi_infos, gt_det_infos


def FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix = 'det_roi_infos.json', 
                        label_map = {1: 0, 2:0, 3:1, 4:2}):
    valid_class = list(label_map.keys())

    detroi_by_case = [] 
    gtroi_by_case = []

    for cid, cid_infos in pid2niifp_map.items():
        det_fp = osp.join(nii_save_dir, f'{cid}_{suffix}')
        if not osp.exists(det_fp): continue
        # gt_fp = cid_infos['roi']
        # gt_det_infos = load2json(gt_fp)
        roi_fp, seg_gt_fp = cid_infos['roi'], cid_infos['label']
        gt_det_infos = []
        # 1. prepare the detection target and segmentation target from either json file and mask labels
        if isinstance(roi_fp, (str, Path)) and osp.exists(roi_fp):
            gt_det_infos = load2json(roi_fp)
        else:
            print(f'[FROC] gt det fp {roi_fp} not exist, please check')

        # if len(gt_det_infos) == 0 and isinstance(seg_gt_fp, (str, Path)) and osp.exists(seg_gt_fp):
        #     gt_roi_mask, af_mat = IO4Nii.read(seg_gt_fp, axis_order= None, verbose = False, dtype=np.uint8)
        #     print_tensor('\tGTmask', gt_roi_mask)
        #     gt_det_infos = roi_info_mask(gt_roi_mask)

        # print(f'{cid} {gt_fp} {det_fp}')
        det_roi_infos = load2json(det_fp)
        detroi_by_case.append(det_roi_infos)
        gtroi_by_case.append(gt_det_infos)
    
    print(f'[ScanPred] {suffix} files ', len(detroi_by_case))
    print('[ScanPred] gt label files ', len(gtroi_by_case))
    safe_nx7_array = lambda x : np.array(x) if len(x) > 0 else np.zeros((0, 7))
    # safe_nx6_array = lambda x : np.array(x) if len(x) > 0 else np.zeros((0, 6))
    gt_bboxes_nx7 = [safe_nx7_array([roi['bbox'] + [roi['class']] for roi in rois if roi['class'] in valid_class]) 
                                for i, rois in enumerate(gtroi_by_case)]
    pred_bbox_nx7 = [safe_nx7_array([roi['bbox'] + [roi['prob']] for roi in rois]) 
                                    for i, rois in enumerate(detroi_by_case)]
    
    ipdb.set_trace()
    result_dict, fig = calculate_FROC_by_center(gt_bboxes_nx7, pred_bbox_nx7, 
                                                luna_output_format=True, plt_figure=True)
    if fig is not None: fig.savefig(osp.join(nii_save_dir, suffix.replace('.json', '_FROC.pdf')))
    return result_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with zaxis augmentation')

    parser.add_argument('--model-name',
                        help='model name',
                        default= 'fcn_hr18_3x449x449_100eps_pleura_fp16_sgd_aug_medview',
                        # required=True,
                        type=str)
    parser.add_argument('--split',
                        help="data split ",
                        default='test',
                        type=str
                        # nargs=argparse.REMAINDER
                        )
    parser.add_argument('--repo-rt',
                        help='repo/code root of mmseg',
                        default= '/home/dejuns/git/mmseg4med/work_dirs/AbdoVeinDataset',
                        # required=True,
                        type=str)
    parser.add_argument('--data-rt',
                        help="data rt ",
                        default='/raid/data/lung/公开数据集/lola11/nii',
                        type=str
                        # nargs=argparse.REMAINDER
                        )
    parser.add_argument('--label-rt',
                        help="label root ",
                        default=None,
                        type=str
                        # nargs=argparse.REMAINDER
                        )
    parser.add_argument('--weight-file',
                        help="the filename of the model weight",
                        default = 'latest.pth',
                        type= str
                        # nargs=argparse.REMAINDER
                        ) 
    parser.add_argument('--gpu-ix',
                        help="gpu-ix, just 1",
                        default= 1,
                        type=int
                        # nargs=argparse.REMAINDER
                        )      
    parser.add_argument('--not-ky-style',
                        help="image filename not in ky style, not ending with _0000.nii.gz",
                        action='store_true'
                        )   
    parser.add_argument('--pos-thresh',
                        help=' thresh value used to decide postive prediction from model probability',
                        default= 0.5,
                        type=float
                        )      
    parser.add_argument('--dataset-name',
                        help="which dataset to infer",
                        default='shiwei',
                        type=str
                        )            
    parser.add_argument('--fold-ix',
                        help="which fold of the data to be infered. Using parallel inference for speedup",
                        default=None,
                        type=int
                        )            
    parser.add_argument('--num-fold',
                    help="number of folds to distribute. Using parallel inference for speedup",
                    default=4,
                    type=int
                    )                     
    parser.add_argument('--eval-final',
                        help="flag for testing",
                        action='store_true'
                    )        
    parser.add_argument('--fp16',
                        help="if infer using fp16 mixed precision",
                        action='store_true'
                    )        
    parser.add_argument('--run_pids',
                    help="select pids to run",
                    type=str,
                    nargs='+',
                    # type = list,
                )      
    parser.add_argument('--verbose',
                    action='store_true',
                    help="if verbose",
                )                                    
    args = parser.parse_args()
    for a in vars(args):  print(f'{a}\t{getattr(args, a)}')
    return args

def has_series_4digit(fp):
    stem = Path(fp).stem
    num_chunks = len(stem.split('_'))
    if num_chunks == 3: return True
    else: return False
    
pid_in_path_ky = lambda x : x.split(os.sep)[-1].split('.')[0]
pid_in_path_rb = lambda x : x.split(os.sep)[-1].split('-')[0]  

def sample_list2run(pid2niifp_map, run_pids = None,
                    num_fold = 4, fold_ix = None):
    pid_pool = list(pid2niifp_map)
    pid_pool = sorted(pid_pool)#[:20]
    total_pid = len(pid_pool)
    print(f'Dataset contains {total_pid} pid in total')
    if fold_ix is not None:
        num_per_fold = int(np.ceil(total_pid / num_fold)) #+ (1 if total_pid % num_fold > 0 else 0)
        if fold_ix * num_per_fold < total_pid:
            start = fold_ix * num_per_fold
            end = (fold_ix + 1) * num_per_fold #if fold_ix != num_fold else total_pid
            if fold_ix == num_fold -1: end = total_pid
            print(f'[Fold{fold_ix}/{num_fold}]: start {start} end {end}')
            pid_pool = pid_pool[start: end]
        else: pid_pool = []
    # save_string_list(data_rt/'entire_sub_dir_list.txt', [str(v) for v in pid2niifp_map.values()])
    # [print(i, p) for i, p in enumerate(run_pids)]
    # print(f'Fold{fold_ix}/{num_fold}:  total pids {len(run_pids)}, first 2 {run_pids[:2]}, last 2 {run_pids[-2:]} \n')
    this_run_pids = [p for p in run_pids if p in pid_pool] if bool(run_pids) else pid_pool
    print(f'Fold{fold_ix}/{num_fold}:  total pids {len(this_run_pids)}, first 2 {this_run_pids[:2]}, last 2 {this_run_pids[-2:]} \n')
    
    pid2niifp_map = {p:pid2niifp_map[p] for p in this_run_pids}
    return pid2niifp_map

def load_list_detdj(data_folder:str,  mode = 'test ', 
                    json_filename = 'dataset.json', 
                    exclude_pids = None,
                    key2suffix = {'image': '_image.nii', 
                                  'label': '_instance.nii', 
                                  'roi':'_ins2cls.json'}):
    """

    img_fp = f"{c}_image.nii"
        np.ndarray
    seg_fp = f"{c}_instance.npy"
        np.ndarray
    roi_fp = "{c}_ins2cls.json"

    rois: list[dict(), dict(), ...]
    one_dict: {'instance' : int, start from 1
                'bbox': list(int), e.g. (x1, y1, z1, x2, y2, z2)
                'class': int, start from 1 
                'center' : list(int), (x, y, z)
                'spine_boudnary': list(int), (x, y)
                }
    return 
        file_list : [{'image' : img_path, 'label' : label_path}, ...]
    """
    file_list = list(), list()
    # a = [print(self.map_key(k)) for k in keys]
    js_fp = os.path.join(data_folder, json_filename)
    if not osp.exists(js_fp):
        print(f'[FileError] {js_fp} not exist') 
        return file_list
    with open(js_fp, 'r') as load_f:
        load_dict = json.load(load_f)
    case_fns = load_dict['training'] if mode == 'train' else load_dict['test']

    pid2pathpairs = {}
    for ix, part_fn in enumerate(case_fns):
        cid = part_fn.split(os.sep)[-1]
        if ix < 2: print('Check cid', cid)
        if exclude_pids and (cid in exclude_pids): 
            print('Exclude ', cid)
            continue
        this_holder = {'cid': cid}
        for k, suffix in key2suffix.items(): 
            this_holder[k] = osp.join(data_folder, part_fn + suffix)
            if ix < 3: print(f'\t{k}', this_holder[k])
        pid2pathpairs[cid] =  this_holder
    # pathpairs_orderd = sorted(pid2pathpairs, key = lambda x: x['cid']) # TODO: debug
    return pid2pathpairs


def load_list_scan(data_rt = '/data/dejuns/lung_nodule/test_case/check100',
                     label_rt = None, 
                      image_suffix = '_image.nii.gz',
                      label_suffix = '_fracture.nii.gz', 
                      ):
    print('Scanning folder for data list')
    pid2niifp_map = {}
    data_rt = Path(data_rt)
    for f in os.listdir(data_rt):
        if not f.endswith(image_suffix): continue
        pid = f.split(image_suffix)[0]
        pid2niifp_map[pid] = {'cid': pid, 'image':data_rt/f, 'label': None, 'roi': None}

    if label_rt is not None:
        label_rt = Path(label_rt)
        for f in os.listdir(label_rt):
            if not f.endswith(label_suffix): continue
            pid = f.split(label_suffix)[0]
            pid2niifp_map[pid]['label'] = label_rt/f
            roi_fn = f.replace(label_suffix, '_ins2cls.json')
            roi_fp = label_rt/roi_fn
            if not osp.exists(roi_fp):
                print(f'[ScanList] {roi_fp}  not exist so generate')
                gt_roi_mask, af_mat = IO4Nii.read(label_rt/f, axis_order= None, verbose = False, dtype=np.uint8)
                print_tensor('\tGTmask', gt_roi_mask)
                gt_det_infos = roi_info_mask(gt_roi_mask)
                save2json(gt_det_infos, label_rt, roi_fn, indent=None, sort_keys=False)
            pid2niifp_map[pid]['roi'] = roi_fp

    print(f'Under {data_rt} {len(pid2niifp_map)} {image_suffix} files found')
    return pid2niifp_map


if __name__ == '__main__':

    # data path preparation 
    cfg = parse_args()
    # print(cfg.model, cfg.run_pid_ixs)
    dataset_name = cfg.dataset_name
    # assert dataset_name in ('shiwei', 'keya', 'rfmix', 'testcase', 'ircad', 'swnew')
    assert isinstance(cfg.run_pids, (type(None), list, tuple))
    if cfg.label_rt:
        pid2niifp_map = load_list_scan(cfg.data_rt, label_rt=cfg.label_rt)
    else:
        pid2niifp_map = load_list_detdj(cfg.data_rt, mode = cfg.split, json_filename = 'dataset.json')
    
    if not cfg.eval_final:
        pid2niifp_map = sample_list2run(pid2niifp_map, run_pids=cfg.run_pids, 
                                                num_fold=cfg.num_fold, 
                                                fold_ix=cfg.fold_ix)
    fold_ix_str = f'{cfg.fold_ix}@{cfg.num_fold}'

    git_rt = Path(cfg.repo_rt)
    model_store_dir = git_rt/cfg.model_name
    pos_thresh_str = str(cfg.pos_thresh).replace('.', '-')
    store_name =  str(f'visual_{cfg.dataset_name}_{cfg.split}_cutoff{cfg.pos_thresh}_nii')
    nii_save_dir = model_store_dir/store_name
    mkdir(nii_save_dir)

    if cfg.eval_final:
        det_suffix, seg_suffix = 'det_roi_infos.json', 'seg_roi_infos.json'
        if cfg.fp16: det_suffix, seg_suffix = f'fp16_{det_suffix}', f'fp16_{seg_suffix}'
        det_froc = FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix=det_suffix)
        seg_froc = FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix=seg_suffix)
        
        print(f'[{cfg.model_name}] Detection FROC \n', det_froc)
        print(f'[{cfg.model_name}] Segmentation FROC \n', seg_froc)

    else:
        main(cfg, 
            nii_save_dir = nii_save_dir, 
            pid2niifp_map=pid2niifp_map,
            fold_ix_str = fold_ix_str,
            # is_test=True
            )

    #  python 

    # num_slices = [512, 1024]

    # for n in num_slices:
    #     sample_ltmemo(memo_store_dir = 'work_dirs/PancreasDataset/long_term_memory', 
    #                 num_ltm_samples = n, point4keep = 8, use_entropy = False, seed = 42)

    # ltm_neg_fp = osp.join(memo_store_dir, 'long_term_memory_neg_%d_by%s.pt' % (point4keep, select_base))