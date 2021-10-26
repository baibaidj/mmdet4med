from skimage.measure import regionprops
from mmdet.datasets.transform4med.io4med import *
import cc3d, os, pdb

from mmdet.utils.this_utils import run_parralel

def load_list_scan(data_rt = '/data/dejuns/lung_nodule/test_case/check100',
                     label_rt = None, extract_roi_info = True, 
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
            if not osp.exists(roi_fp) and extract_roi_info:
                print(f'[ScanList] {roi_fp}  not exist so generate')
                gt_roi_mask, af_mat = IO4Nii.read(label_rt/f, axis_order= None, verbose = False, dtype=np.uint8)
                print_tensor('\tGTmask', gt_roi_mask)
                gt_det_infos = roi_info_mask(gt_roi_mask)
                save2json(gt_det_infos, label_rt, roi_fn, indent=None, sort_keys=False)
            pid2niifp_map[pid]['roi'] = roi_fp

    print(f'Under {data_rt} {len(pid2niifp_map)} {image_suffix} files found')
    return pid2niifp_map

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


def largest_region_index(mask_multi):
    non_zero_arr = mask_multi[mask_multi > 0].flatten()
    if non_zero_arr.size == 0: return 0
    mask_max_index = np.argmax(np.bincount(non_zero_arr))
    return int(mask_max_index)

def roi_info_mask(mask_semantic, is_prob = False, is_percent = False):
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
        roi_vol = roi_prop['area']
        # print(f'[Roi] ix {ix} vol {roi_vol}')
        if is_prob: 
            roi_cls = 0
        else:
            roi_cls = int(largest_region_index(mask_semantic[roi_slicer]))
        roi_prob = roi_prop.mean_intensity / (100 if is_percent else 1)
        roi_info = {'instance' : ix + 1, 'bbox': roi_bbox, 
                    'class': -1 if roi_cls == 66535 else roi_cls , 
                    'center' : roi_center, 'prob': float(roi_prob)}
        case_roi_infos.append(roi_info)
    return case_roi_infos

def compute_hitmap_1case(pred_roi_infos, gt_roi_infos, verbose = False):
    
    gt_bbox_kx6 = np.array([r['bbox']  for r in gt_roi_infos])
    pred_bbox_nx7 = np.array([r['bbox'] + [r['prob']] for r in pred_roi_infos])

    if gt_bbox_kx6.size > 0:
        gt_bbox_kx2x3 = gt_bbox_kx6.reshape(-1, 2, 3) 
    else: 
        gt_bbox_kx2x3 = np.zeros((0, 2, 3))
    
    if pred_bbox_nx7.size == 0:
        pred_bbox_nx7 = np.zeros((0, 7))

    gt_k, pred_n = int(gt_bbox_kx2x3.shape[0]), int(pred_bbox_nx7.shape[0])

    pred2gt_hitmap = np.zeros((pred_n, gt_k))

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

    return pred2gt_hitmap

def fuse_pred_anno_mask(pid2niifp_map, pred_dir, merge_dir, num_class = 4, maintain_tp = False, prcs_ix = 99):

    for pid, key2fp in pid2niifp_map.items():
        # print(f'Process {pid}')
        roi_fp, seg_gt_fp = key2fp['roi'], key2fp['label']
        seg_fn = str(seg_gt_fp).split(os.sep)[-1]
        pred_fp = pred_dir/str(seg_fn).replace(r'_fracture.nii.gz', r'_image_seg_ribfrac.nii.gz')

        # 1. prepare the detection target and segmentation target from either json file and mask labels
        if osp.exists(seg_gt_fp):
            gt_roi_mask, af_mat = IO4Nii.read(seg_gt_fp, axis_order= None, verbose = False, dtype=np.uint8)
            # print_tensor('\tGTmask', gt_roi_mask)
            gt_det_infos = roi_info_mask(gt_roi_mask)
        else:
            print(f'{seg_gt_fp} not exist, skip')
            continue
        
        if osp.exists(pred_fp):
            pred_roi_mask, af_mat = IO4Nii.read(pred_fp, axis_order=None, verbose=False, dtype=np.uint8)
            pred_det_infos = roi_info_mask(pred_roi_mask)
        else:
            print(f'{pred_fp} not exist, skip')
            continue

        num_roi_pred, num_roi_gt = len(pred_det_infos), len(gt_det_infos)
        pred2gt_hitmap = compute_hitmap_1case(pred_det_infos, gt_det_infos)
        
        merge_mask = np.copy(gt_roi_mask)
        # process false positive by iterating over the row of pred2gt_hitmap. all zero of a row means fp

        fp_pred_ixs = list(range(num_roi_pred))
        if num_roi_gt > 0:
            pred_hit_vector = pred2gt_hitmap.max(axis = 1)
            fp_pred_ixs = np.where(pred_hit_vector < 1)[0]
    
        tp_gt_ixs = []
        if num_roi_pred > 0:
            gt_hit_vector = pred2gt_hitmap.max(axis = 0)
            tp_gt_ixs = np.where(gt_hit_vector == 1)[0]
        # print(pred2gt_hitmap)
        print(f'[Worker{prcs_ix}] pid {pid} FP {fp_pred_ixs} {len(fp_pred_ixs)}/{num_roi_pred};'
                 f'TP {tp_gt_ixs} {len(tp_gt_ixs)}/{num_roi_gt}')

        for ix in fp_pred_ixs:
            roi_info = pred_det_infos[ix]
            roi_bbox, roi_cls = roi_info['bbox'], roi_info['class']
            roi_slicer = tuple([slice(int(roi_bbox[i]), int(roi_bbox[i+3]))  for i in range(3)])
            merge_mask[roi_slicer] = (pred_roi_mask[roi_slicer] > 0) * (num_class + 1)
        
        if not maintain_tp:
            for ix in tp_gt_ixs:
                roi_info = gt_det_infos[ix]
                roi_bbox, roi_cls = roi_info['bbox'], roi_info['class']
                roi_slicer = tuple([slice(int(roi_bbox[i]), int(roi_bbox[i+3]))  for i in range(3)])
                merge_mask[roi_slicer] = (gt_roi_mask[roi_slicer] > 0) * (num_class + 2)
            
        merge_fn = seg_fn.replace('_fracture.nii.gz', '_fracture_refine')
        IO4Nii.write(merge_mask, merge_dir, merge_fn, af_mat)


if __name__ == '__main__':

    # merge scheme: 
    #   FN: (1, 2, 3, 4) 
    #   FP: 5
    #   TP: 6
    num_class, is_maintain_tp = 4, False
    cv_dir, cv_version = 'cv02_test_515', '2.2.5'
    # cv_dir, cv_version = 'cv12_test_516', '2.2.6'

    merge_name = 'man_ai_mask' + ('_fpr' if is_maintain_tp else '')
    cvfold_rt = Path(f'/mnt/data4t/dejuns/ribfrac/infer4refine/{cv_dir}')
    pred_dir = cvfold_rt/f'pred_v{cv_version}'
    merge_dir = cvfold_rt/merge_name # reduce false positive

    mkdir(merge_dir)
    pid2niifp_map_all = load_list_scan(data_rt=cvfold_rt/'image',  label_rt = cvfold_rt/'mask', extract_roi_info=False)
    pids = sorted(list(pid2niifp_map_all)) #[:4]
    pids = ['1637531-20200917'] # failed in 2.2.6 1648785-20201007
    # pids = ['1676858-20201115', '1627860-20200904', '1644669-20200927'] #failed in 2.2.5

    pid2niifp_map = {k:pid2niifp_map_all[k] for k in pids}
    run_parralel(fuse_pred_anno_mask, pid2niifp_map, 
                pred_dir,  merge_dir,  
                num_class = num_class, 
                maintain_tp = is_maintain_tp, 
                num_workers = 1)


    # /mnt/data4t/dejuns/ribfrac/infer4refine/cv12_test_516/pred_v2.2.6/1648785-20201007_image_seg_ribfrac.nii.gz not exist??