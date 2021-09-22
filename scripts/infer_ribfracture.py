
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
import cc3d
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
        # if cid != '1112514_20180429': continue  #'1214845_20181014' 
        print(f'\n[INFER] {pcount} {img_nii_fp}')
        if is_test: continue
        # store_dir
        if cfg.fp16: cid_infos['cid'] = f'{cid}_fp16'
        img_3d, affine_matrix_i = IO4Nii.read(img_nii_fp, axis_order = None, 
                                                verbose = False, dtype=np.int16)
        cid_infos['image_3d'] = img_3d
        cid_infos['affine'] = affine_matrix_i
        timer = Timer() 
        det_results, seg_results = inference_detector4med(model, img_3d, 
                                                        affine = affine_matrix_i,
                                                        rescale = True,
                                                        need_probs = True)
        seg_results = seg_results.numpy()                                            
        torch.cuda.empty_cache()
        if cfg.verbose: print_tensor('\tseg logits', seg_results)
        duration  = timer.since_start()
        
        (case_result, det_roi_infos, seg_roi_infos, gt_det_infos) = \
                evaluate_store_prediction_1case(det_results, seg_results,
                                                cid_infos, nii_save_dir, 
                                                seg_prob_thresh=cfg.pos_thresh)
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

def evaluate_store_prediction_1case(det_results, seg_results, cid_infos, nii_save_dir, seg_prob_thresh = 0.5):
    """
    Args:
        det_results: list of detection results of 1 image
        seg_results: 5d tensor
    """
    

    cid, img_3d, affine_mat = cid_infos['cid'], cid_infos['image_3d'], cid_infos['affine']

    det_roi_infos = roi_info_dets(det_results[0])
    seg_roi_prob = post_process_ribfrac(seg_results[0], image_spine_2d=locate_spine_region(img_3d), 
                                        prob_thresh=seg_prob_thresh)
    # pdb.set_trace()
    seg_roi_infos = roi_info_mask(seg_roi_prob,  is_prob = True)
    print_tensor('\timage', img_3d)
    print_tensor('\tpred', seg_results)


    gt_det_infos = load2json(cid_infos['roi'])
    # recall, precision, fp, fn, tp
    metric_keys = ('recall', 'precision', 'fp', 'fn', 'tp')
    det_roi_metrics = evaluate1case(det_roi_infos, gt_det_infos)
    seg_roi_metrics = evaluate1case(seg_roi_infos, gt_det_infos)
    det_result = {f'det_{k}': det_roi_metrics[i]  for i, k in  enumerate(metric_keys)}
    seg_result = {f'seg_{k}': seg_roi_metrics[i]  for i, k in  enumerate(metric_keys)}
    
    seg_gt_fp = cid_infos['label']
    if seg_gt_fp: 
        gt_roi_mask, af_mat = IO4Nii.read(seg_gt_fp, axis_order= None, verbose = False, dtype=np.uint8)
        print_tensor('\tGTmask', gt_roi_mask)
        gt_roi_mask = np.where(gt_roi_mask > 0, 1, 0) 
        # if label_mapping is not None: gt_mask = convert_label(gt_mask, label_mapping = label_mapping, value4outlier=1)
    else: gt_roi_mask = None
    seg_roi_mask = np.array(seg_roi_prob > 0, dtype = np.uint8)
    dice3d = evaluate_pred_against_label(seg_roi_mask, gt_roi_mask, ('dice', ), 2)['dice']

    # pdb.set_trace()
    det_roi_mask = np.zeros_like(img_3d, dtype = np.uint8)
    for roi in det_roi_infos:
        # slicer = tuple([slice(roi['bbox'][i], roi['bbox'][i + 3])  for i in range(3)])
        bbox2slicer = lambda b: tuple([slice(round(b[i]), round(b[i + 3]))  for i in range(3)])
        slicer = bbox2slicer(roi['bbox'])
        det_roi_mask[slicer] = round(roi['prob'] * 100)

    IO4Nii.write(seg_roi_mask, nii_save_dir, f'{cid}_roi_seg_mask', affine_mat)
    IO4Nii.write(det_roi_mask, nii_save_dir, f'{cid}_roi_det_mask', affine_mat)

    save2json(det_roi_infos, nii_save_dir, f'{cid}_det_roi_infos.json', indent=None, sort_keys=False)
    save2json(seg_roi_infos, nii_save_dir, f'{cid}_seg_roi_infos.json', indent=None, sort_keys=False)

    img_dim = {f'img_dim{i}': img_3d.shape[i] for i in range(3)}
    space_dim = {f'spacing_dim{i}': affine_mat[i, i] for i in range(3)}
    case_result = {'cid' : cid, **img_dim, **space_dim,
                    **det_result, **seg_result, 'seg_dice': float(dice3d)}
    return case_result, det_roi_infos, seg_roi_infos, gt_det_infos


def FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix = 'det_roi_infos.json'):
    detroi_by_case = [] 
    gtroi_by_case = []

    for cid, cid_infos in pid2niifp_map.items():
        det_fp = osp.join(nii_save_dir, f'{cid}_{suffix}')
        if not osp.exists(det_fp): continue
        gt_fp = cid_infos['roi']
        gt_det_infos = load2json(gt_fp)
        # print(f'{cid} {gt_fp} {det_fp}')
        det_roi_infos = load2json(det_fp)
        detroi_by_case.append(det_roi_infos)
        gtroi_by_case.append(gt_det_infos)
    
    safe_nx7_array = lambda x : np.array(x) if len(x) > 0 else np.zeros((0, 7))
    safe_nx6_array = lambda x : np.array(x) if len(x) > 0 else np.zeros((0, 6))

    gt_bboxes = [safe_nx6_array([roi['bbox'] for roi in rois]) for i, rois in enumerate(gtroi_by_case)]
    pred_bbox_nx7 = [safe_nx7_array([roi['bbox'] + [roi['prob']] for roi in rois]) for i, rois in enumerate(detroi_by_case)]
    result_dict, fig = calculate_FROC_by_center(gt_bboxes, pred_bbox_nx7, 
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
    parser.add_argument('--run_pid_ixs',
                    help="select pids to run",
                    type=int,
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

def sample_list2run(pid2niifp_map, run_pid_ixs = None,
                    num_fold = 4, fold_ix = None):
    run_pids = list(pid2niifp_map)
    run_pids = sorted(run_pids)#[:20]
    total_pid = len(run_pids)
    print(f'Dataset contains {total_pid} pid in total')
    if fold_ix is not None:
        num_per_fold = int(np.ceil(total_pid / num_fold)) #+ (1 if total_pid % num_fold > 0 else 0)
        if fold_ix * num_per_fold < total_pid:
            start = fold_ix * num_per_fold
            end = (fold_ix + 1) * num_per_fold #if fold_ix != num_fold else total_pid
            if fold_ix == num_fold -1: end = total_pid
            print(f'[Fold{fold_ix}/{num_fold}]: start {start} end {end}')
            run_pids = run_pids[start: end]
        else: run_pids = []
    # save_string_list(data_rt/'entire_sub_dir_list.txt', [str(v) for v in pid2niifp_map.values()])
    # [print(i, p) for i, p in enumerate(run_pids)]
    # print(f'Fold{fold_ix}/{num_fold}:  total pids {len(run_pids)}, first 2 {run_pids[:2]}, last 2 {run_pids[-2:]} \n')
    run_pids = [run_pids[i] for i in run_pid_ixs] if bool(run_pid_ixs) else run_pids
    print(f'Fold{fold_ix}/{num_fold}:  total pids {len(run_pids)}, first 2 {run_pids[:2]}, last 2 {run_pids[-2:]} \n')
    
    pid2niifp_map = {p:pid2niifp_map[p] for p in run_pids}
    return pid2niifp_map

def load_list_detdj(data_folder:str,  mode = 'test ', 
                    json_filename = 'dataset.json', 
                    exclude_pids = None,
                    key2suffix = {'image': '_image.nii.gz', 
                                  'label': '_instance.nii.gz', 
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
    if not osp.exists(js_fp): return file_list
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


if __name__ == '__main__':

    # data path preparation 
    cfg = parse_args()
    # print(cfg.model, cfg.run_pid_ixs)
    dataset_name = cfg.dataset_name
    # assert dataset_name in ('shiwei', 'keya', 'rfmix', 'testcase', 'ircad', 'swnew')
    assert isinstance(cfg.run_pid_ixs, (type(None), list, tuple))

    pid2niifp_map = load_list_detdj(Path(cfg.data_rt), mode = cfg.split)
    
    if not cfg.eval_final:
        pid2niifp_map = sample_list2run(pid2niifp_map, run_pid_ixs=cfg.run_pid_ixs, 
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
        det_froc = FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix='det_roi_infos.json')
        seg_froc = FROC_dataset_level(pid2niifp_map, nii_save_dir, suffix='seg_roi_infos.json')
        
        print('Detection FROC', det_froc)
        print('Segmentation FROC', seg_froc)

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