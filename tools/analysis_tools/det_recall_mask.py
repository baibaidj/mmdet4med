import cc3d, os
import numpy as np
import os.path as osp
from skimage.measure import regionprops
from mmdet.datasets.transform4med.io4med import IO4Nii
from mmdet.utils.this_utils import run_parralel
import pandas as pd



def largest_region_index(mask_multi):
    non_zero_arr = mask_multi[mask_multi > 0].flatten()
    if non_zero_arr.size == 0: return 0
    mask_max_index = np.argmax(np.bincount(non_zero_arr))
    return int(mask_max_index)

def roi_info_mask(mask_semantic):
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
        roi_cls = int(largest_region_index(mask_semantic[roi_slicer]))
        roi_info = {'instance' : ix + 1, 'bbox': roi_bbox, 'class': -1 if roi_cls == 66535 else roi_cls , 
                    'center' : roi_center}
        case_roi_infos.append(roi_info)
    return case_roi_infos

def evaluate1case(pred_roi_infos, gt_roi_infos):
    
    gt_bbox_kx2x3 = np.array([r['bbox'] for r in gt_roi_infos]).reshape(-1, 2, 3)
    pred_point_nx3 = np.array([r['center'] for r in pred_roi_infos])
    gt_k, pred_n = gt_bbox_kx2x3.shape[0], pred_point_nx3.shape[0]

    point2bbox_start = pred_point_nx3[:, None, :] -  gt_bbox_kx2x3[None, :, 0]  # nx1x3 - 1xkx3, nxkx3
    point2bbox_end = -1 * pred_point_nx3[:, None, :] + gt_bbox_kx2x3[None, :, 1] 
    point2bbox_nxkx6 = np.stack([point2bbox_start, point2bbox_end], axis = -1).reshape(pred_n, gt_k, 6)
    pred2gt_nxk = np.amin(point2bbox_nxkx6, axis = -1)  >= 0 
    
    hit_count_k =  np.sum(pred2gt_nxk, axis=0)
    tp = np.sum(hit_count_k)
    fp = pred_n - tp
    fn = gt_k - np.sum(hit_count_k > 0)
    recall = tp / gt_k
    precision = tp/ pred_n
    return recall, precision, fp, fn

def extract_roi_info_bunch(pids, pid2pairs):
    pid_info_list = []
    for pid in pids:
        pair_dict = pid2pairs[pid]
        pred_fp, gt_fp = pair_dict['pred_fp'], gt_fp['gt_fp']
        pred_mask, af = IO4Nii.read(pred_fp)
        gt_mask, af = IO4Nii.read(gt_fp)
        pred_rois = roi_info_mask(pred_mask)
        gt_rois = roi_info_mask(gt_mask)
        pid_info = {'pid' : pid, 'gt_rois': gt_rois, 'pred_rois': pred_rois}
        pid_info_list.append(pid_info)
    return pid_info_list

def compuate_det_metric_bunch(pid_info_list):
    for infos in pid_info_list:
        gt_rois  = infos['gt_rois']
        pred_rois = infos['pred_rois']
        recall, precision, fp, fn = evaluate1case(pred_rois, gt_rois)
        this_result = {'recall': recall, 'precision': precision, 
                        'fp': fp, 'fn' : fn}
        infos.update(this_result)
    return pid_info_list


def scan4path_pairs(pred_dir, gt_dir):
    pid2preds = {f.split('_')[0]: osp.join(pred_dir, f) for f in os.listdir(pred_dir)}
    pid2gts = {f.split('_')[0]: osp.join(gt_dir, f) for f in os.listdir(gt_dir)}

    pid2pairs = {}
    for pid, ofp in pid2preds.items():
        if pid not in pid2gts:
            continue
        else:
            pid2pairs.setdefault(pid, {'pred_fp' : ofp, 'gt_fp': pid2gts[pid]})
    
    print(f'[Scan] gt {len(pid2gts)} pred {len(pid2gts)}  matched {len(pid2pairs)}')
    return pid2pairs

def main(pred_dir, gt_dir, save_dir, csv_fn = 'det_evaluation_result.csv'):

    pid2pairs = scan4path_pairs(pred_dir, gt_dir)
    pids = list(pid2pairs)[:8]
    pid_info_list = run_parralel(extract_roi_info_bunch, pids, pid2pairs, num_workers=4)
    pid_info_list = run_parralel(compuate_det_metric_bunch, pid_info_list, num_workers=4)

    result_tb = pd.DataFrame(pid_info_list)
    summary_tb = result_tb.describe()
    print(summary_tb)
    result_tb.to_csv(osp.join(save_dir, csv_fn))

if __name__ == '__main__' :

    pred_dir = '/data/dejuns/ribfrac/processed/organize_raw/origin_labels'
    gt_dir = '/data/dejuns/ribfrac/processed/organize_raw/refine_labels'
    save_dir = '/data/dejuns/ribfrac/processed/organize_raw'
    
    main(pred_dir, gt_dir, save_dir, csv_fn = 'check_refine_annotation_frac.csv')

    

# for i in */*rib*;do sshpass -p "90920112" rsync -vzuLPR /raid/chengweis/ribfrac/raw_data/50_result/./$i dejuns@10.3.6.81:/data/dejuns/ribfrac/raw_rename/KY_B4_50;done