import re
import cc3d, os, pdb
import numpy as np
import os.path as osp
from skimage.measure import regionprops
from mmdet.datasets.transform4med.io4med import IO4Nii, print_tensor, save2json, load2json
from mmdet.utils.this_utils import  run_parralel
import pandas as pd



def largest_region_index(mask_multi):
    non_zero_arr = mask_multi[mask_multi > 0].flatten()
    if non_zero_arr.size == 0: return 0
    mask_max_index = np.argmax(np.bincount(non_zero_arr))
    return int(mask_max_index)

def roi_info_mask(mask_semantic, is_prob = False):
    mask_instance, num_rois = cc3d.connected_components(
                                mask_semantic > 0, return_N = True)
    if num_rois == 0:
        return []
    instance_props = regionprops(mask_instance, mask_semantic)
    case_roi_infos = []
    for ix, roi_prop in enumerate(instance_props):
        roi_bbox = [round(a) for a in roi_prop['bbox']] #np.array(prop['bbox'])
        roi_center = [round(a) for a in roi_prop['centroid']]
        roi_slicer = roi_prop.slice #tuple([slice(int(roi_bbox[i]), int(roi_bbox[i+3]))  for i in range(3)])
        if is_prob: 
            roi_cls = 0
        else:
            roi_cls = int(largest_region_index(mask_semantic[roi_slicer]))
        roi_prob = roi_prop.mean_intensity
        roi_info = {'instance' : ix + 1, 'bbox': roi_bbox, 
                    'class': -1 if roi_cls == 66535 else roi_cls , 
                    'center' : roi_center, 'prob': float(roi_prob)}
        case_roi_infos.append(roi_info)
    return case_roi_infos


def roi_info_dets(det_results):
    """
    Args:
        the detection result of one image
        [list[np.ndarray, nx7],  ] classes, bbox_nx7
    
    return

    """
    case_roi_infos = []
    for cls_i, bbox_nx7 in enumerate(det_results):
        bbox_nx2x3 = bbox_nx7[:, :6].reshape(-1, 2, 3)
        roi_center_nx3 = bbox_nx2x3.mean(axis = 1)
        for bi, bnx7 in enumerate(bbox_nx7):
            roi_bbox = [float(a) for a in bnx7[:6]] 
            roi_center = [float(a) for a in roi_center_nx3[bi]]
            roi_info = {'instance' : bi + 1, 'bbox': roi_bbox, 
                        'class': cls_i, 'center': roi_center, 
                        'prob': float(bnx7[-1])}
            case_roi_infos.append(roi_info)
    
    case_roi_infos = sorted(case_roi_infos, key = lambda x: x['prob'], reverse=True)
    return case_roi_infos


def point2bbox_hitmap(pred_point_nx3, gt_bbox_kx2x3):

    assert pred_point_nx3.shape[-1] == 3
    assert gt_bbox_kx2x3.shape[-2:] == (2, 3)

    pred_n, gt_k = pred_point_nx3.shape[0], gt_bbox_kx2x3.shape[0]

    if gt_k == 0 or pred_n == 0:
        pred2gt_nxk = np.zeros((pred_n, gt_k)) - 1
    else:
        point2bbox_start = pred_point_nx3[:, None, :] -  gt_bbox_kx2x3[None, :, 0]  # nx1x3 - 1xkx3, nxkx3
        point2bbox_end = -1 * pred_point_nx3[:, None, :] + gt_bbox_kx2x3[None, :, 1]  # 
        point2bbox_nxkx6 = np.stack([point2bbox_start, point2bbox_end], axis = -1).reshape(pred_n, gt_k, 6)
        pred2gt_nxk = np.amin(point2bbox_nxkx6, axis = -1)  >= 0 # nxk
    return pred2gt_nxk

def evaluate1case(pred_roi_infos, gt_roi_infos, verbose = False):
    
    gt_bbox_kx6 = np.array([r['bbox']  for r in gt_roi_infos])
    # gt_point_kx3 = np.array([r['center'] for r in gt_roi_infos])
    pred_bbox_nx7 = np.array([r['bbox'] + [r['prob']] for r in pred_roi_infos])

    if gt_bbox_kx6.size > 0:
        gt_bbox_kx2x3 = gt_bbox_kx6.reshape(-1, 2, 3) 
    else: 
        gt_bbox_kx2x3 = np.zeros((0, 2, 3))
    
    if pred_bbox_nx7.size == 0:
        pred_bbox_nx7 = np.zeros((0, 7))

    gt_k, pred_n = int(gt_bbox_kx2x3.shape[0]), int(pred_bbox_nx7.shape[0])

    if gt_k == 0:
        recall, precision, fp, fn, tp = np.nan, 0, pred_n, 0, 0
    
    if pred_n == 0:
        recall, precision, fp, fn, tp = 0, np.nan, 0, gt_k, 0
    
    if gt_k != 0 and pred_n != 0:
        pred_bbox_nx2x3 = pred_bbox_nx7[:, :6].reshape(-1, 2, 3)
        pred_point_nx3 = pred_bbox_nx2x3.mean(axis = 1)
        gt_point_kx3 = gt_bbox_kx2x3.mean(axis=1)
        pred2gt_nxk = point2bbox_hitmap(pred_point_nx3, gt_bbox_kx2x3)
        gt2pred_kxn = point2bbox_hitmap(gt_point_kx3, pred_bbox_nx2x3)
        if verbose: 
            print_tensor('\tpred2gt', pred2gt_nxk)
            print_tensor('\tgt2pred', gt2pred_kxn)
            # print_tensor('\tPred2gt', pred2gt_hitmap)
        # pdb.set_trace()
        pred2gt_hitmap = np.stack([pred2gt_nxk, gt2pred_kxn.transpose(1, 0)], axis = -1).max(axis = -1)

        # pdb.set_trace()
        hit_count_k =  np.amax(pred2gt_hitmap, axis=0)
        tp = int(np.sum(hit_count_k))
        fp = int(pred_n - tp)
        fn = gt_k - tp
        recall = tp / gt_k
        precision = tp/ pred_n
    return recall, precision, fp, fn, tp


def extract_roi_info_bunch(pids, pid2pairs, verbose = False, **kwargs):
    pid_info_list = []
    for pid in pids:
        print('[LoadRoi] pid', pid)
        pair_dict = pid2pairs[pid]
        pred_fp, gt_fp = pair_dict['pred_fp'], pair_dict['gt_fp']
        pred_mask, af = IO4Nii.read(pred_fp, verbose=verbose, dtype=np.uint8)
        gt_mask, af = IO4Nii.read(gt_fp, verbose= verbose, dtype = np.uint8)
        pred_rois = roi_info_mask(pred_mask)
        gt_rois = roi_info_mask(gt_mask)
        pid_info = {'pid' : pid, 'gt_rois': gt_rois, 'pred_rois': pred_rois}
        pid_info_list.append(pid_info)
    return pid_info_list


def compuate_det_metric_bunch(pid_info_list, verbose = False, **kwargs):

    for infos in pid_info_list:
        print('[Metric] pid', infos['pid'])
        gt_rois  = infos['gt_rois']
        pred_rois = infos['pred_rois']
        recall, precision, fp, fn, tp = evaluate1case(pred_rois, gt_rois)
        this_result = {'recall': recall, 'precision': precision, 
                        'fp': fp, 'fn' : fn, 'tp': tp}
        infos.update(this_result)
        if verbose: print(infos)
    return pid_info_list


def scan4path_pairs(pred_dir, gt_dir):

    pid2preds = {f.split('_')[0]: osp.join(pred_dir, f) for f in os.listdir(pred_dir)}
    pid2gts = {f.split('_')[0]: osp.join(gt_dir, f) for f in os.listdir(gt_dir)}

    pid2pairs = dict()
    for pid, ofp in pid2preds.items():
        if pid not in pid2gts:
            continue
        else:
            pid2pairs.setdefault(pid, {'pred_fp' : ofp, 'gt_fp': pid2gts[pid]})
    
    print(f'[Scan] gt {len(pid2gts)} pred {len(pid2gts)}  matched {len(pid2pairs)}')
    return pid2pairs


def main_eval(pred_dir, gt_dir, save_dir, store_fn = 'det_evaluation_result'):

    json_store_fp = osp.join(save_dir, store_fn + '.json')

    if osp.exists(json_store_fp):
        print('Pid info read from ', json_store_fp)
        pid_info_list = load2json(json_store_fp)
    else:
        pid2pairs = scan4path_pairs(pred_dir, gt_dir)
        pids_all = sorted(list(pid2pairs), key = lambda k: int(k[7:])) #[:2]
        pid2run = pids_all #['RibFrac332']#
        pid_info_list, *_ = run_parralel(extract_roi_info_bunch, pid2run, pid2pairs, num_workers=8)

    save2json(pid_info_list, save_dir, store_fn + '.json', indent=False, sort_keys=False)
    print('done extracing rois')
    pid_info_list, *_ = run_parralel(compuate_det_metric_bunch, pid_info_list, num_workers=8)
    print('done computing metrics')
    for info in pid_info_list:
        info.pop('gt_rois')
        info.pop('pred_rois')
    result_tb = pd.DataFrame(pid_info_list)
    summary_tb = result_tb.describe()
    print(summary_tb)
    result_tb.to_csv(osp.join(save_dir, store_fn + '.csv'), index = False)
    
    tp_total = np.sum(result_tb['tp'])
    fp_total = np.sum(result_tb['fp'])
    fn_total = np.sum(result_tb['fn'])
    gt_total = (tp_total + fn_total)
    recall_total = tp_total / gt_total
    precision_total = tp_total / (tp_total + fp_total)

    print(f'Per Lesion Metric : count {gt_total}  recall {recall_total}  precision {precision_total}')



from mmdet.core.evaluation.mean_dice import *

def evaluate_pred_against_label(pred_3d, gt_3d = None, metric_keys = ('dice', ), num_classes = 3):
    
    if gt_3d is None: gt_none_result = [np.nan for i in range(1, num_classes)]
    else: gt_none_result = None 
    metric2results = {}
    for metric_key in metric_keys:
        if gt_none_result is not None: this_result_by_class = gt_none_result
        else:
            if metric_key == 'dice':
                npround = lambda x: np.around(x, 4)
                cfs_matrix_list = cfsmat4mask_batched(pred_3d, gt_3d, num_classes)
                metric2ds, metric3d = metric_in_cfsmat_1by1(cfs_matrix_list)
                # suppose there are c classes
                dice2d_e = npround(np.array([a['dice'] for a in metric2ds]).mean(0)) # [d1, d2, ...dc]
                dice3d_e = npround(metric3d['dice']) # [d1, d2, ...dc]
                # print('Check dice', dice3d_e)
                this_result_by_class = dice3d_e[1:]
            else: 
                raise NotImplementedError (f'metric keys can only be one of three dice, sfd and hdf but get {metric_key}')
        # print('CH')
        metric2results[metric_key] = this_result_by_class

    return metric2results


if __name__ == '__main__' :

    pred_dir = '/data/dejuns/ribfrac/processed/organize_raw/origin_labels'
    gt_dir = '/data/dejuns/ribfrac/processed/organize_raw/refine_labels'
    save_dir = '/data/dejuns/ribfrac/processed/organize_raw'
    
    main_eval(pred_dir, gt_dir, save_dir, store_fn = 'check_refine_annotation_frac')

    
