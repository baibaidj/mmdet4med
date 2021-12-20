from .io_helper import * 

def load_list_scan(data_rt = '/data/dejuns/lung_nodule/test_case/check100',
                     label_rt = None, extract_roi_info = True, 
                      image_suffix = '_image.nii.gz',
                      label_suffix = '_fracture.nii.gz', 
                      ):
    print('Scanning folder for data list')
    pid2niifp_map = {}
    if data_rt is not None and osp.exists(data_rt):
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
            pid2niifp_map.setdefault(pid, {'cid': pid, 'image':None, 'label': None, 'roi': None})
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

def fuse_pred_anno_mask(pid2niifp_map, pred_dir, merge_dir, 
                        label_suffix = '_fracture.nii.gz', 
                        pred_suffix = '_image_seg_ribfrac.nii.gz', 
                        num_class = 4, maintain_tp = False, 
                        prcs_ix = 99):

    for pid, key2fp in pid2niifp_map.items():
        # print(f'Process {pid}')
        roi_fp, seg_gt_fp = key2fp['roi'], key2fp['label']
        seg_fn = str(seg_gt_fp).split(os.sep)[-1]
        pred_fp = pred_dir/str(seg_fn).replace(label_suffix, pred_suffix)

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
                 f'\tTP {tp_gt_ixs} {len(tp_gt_ixs)}/{num_roi_gt}')

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


def get_fracture_info(det_frac_nx8, rib_instance_mask, rib_parts_mask, affine_matrix_raw):
    """
    
    """
    value2name_ribcls = {1:'displaced', 2: 'nondisplaced', 3: 'buckle', 4: 'old', 5: 'FP', 7: 'others'}
    value2name_ribside = {1: 'L', 0: 'R'} # 1 is odd, 0 is even
    value2name_ribparts = {1: 'front', 2: 'middle', 3:'back'}
    v6_xyz2zyx = lambda x: x[[2, 1, 0, 5, 4, 3]]

    assert rib_instance_mask.shape == rib_parts_mask.shape
    # frac_instance_mask, num_fracs = cc3d.connected_components(frac_semantic_mask > 0, return_N=True)
    # frac_instance_props = regionprops(frac_instance_mask)
    affine_matrix = affine_matrix_raw.copy()
    for i in range(3): affine_matrix[i, i] = abs(affine_matrix[i, i])
    frac_proposal_info = []
    det_ix2cls = {}
    valid_ix = 1
    for i, roi_v8 in enumerate(det_frac_nx8):
        roi_bbox_xyz = roi_v8[:6].astype(np.int16) #x0y0z0 x1y1z1
        roi_bbox_zyx = v6_xyz2zyx(roi_bbox_xyz)
        # prop_center = tuple([int(a) for a in prop['centroid']])
        roi_slicer = tuple([slice(int(roi_bbox_zyx[i]), int(roi_bbox_zyx[i+3]))  for i in range(3)])
        roi_cls = roi_v8[-1] #largest_region_index(frac_semantic_mask[prop_slicer])
        roi_loc_rib = largest_region_index(rib_instance_mask[roi_slicer])
        roi_loc_rib_part = largest_region_index(rib_parts_mask[roi_slicer])
        # roi_loc_spine = largest_region_index(spine_sementic_mask[roi_slicer])
        if roi_loc_rib == 0: 
            det_ix2cls[i] = (0, 0)
            continue
        roi_bbox_px = np.concatenate([roi_bbox_xyz.reshape(2, 3).transpose(), np.ones((1, 2))], axis = 0)# 3x2> 4x2
        roi_bbox_world = np.dot(affine_matrix, roi_bbox_px).astype(float) #.astype(np.int32) # 4x4  4x2
        roi_bbox_world[:2] *= -1  # 4x2, xyz1; ikt-> world, reverse the sign of xy
        
        roi_bbox_world = list(roi_bbox_world[:3, 0]) + list(roi_bbox_world[:3, 1])
        frac_info = {'mask': valid_ix, 
                    'fracClass': value2name_ribcls.get(roi_cls, 'NA'), 
                    'ribSide': value2name_ribside.get(roi_loc_rib%2, 'NA'), 
                    'ribNum': int(roi_loc_rib//2 + (1 if roi_loc_rib%2 > 0 else 0)), 
                    'ribType': value2name_ribparts.get(roi_loc_rib_part, 'NA'), 
                    'ribPosition': roi_bbox_world, 
                    'rib_bbox_px': [int(a) for a in roi_bbox_xyz], 
                    'fracProb': round(roi_v8[6], 6), 
                    }
        frac_proposal_info.append(frac_info)
        det_ix2cls[i] = (valid_ix, roi_cls + 1)
        valid_ix += 1

    return frac_proposal_info, det_ix2cls

def profile_fracture_loops( pid2niifp_map, 
                            structure_dir = '', 
                            instance_suffix = '_image_seg_ribs_instance.nii.gz', 
                            region_suffix = '_image_seg_rib_types.nii.gz', 
                            cohort_id = '', 
                            prcs_ix = 99, 
                            ):
    """
    

    Args:
        lesion_dir (str, optional): [description]. Defaults to ''.
        structure_dir (str, optional): [description]. Defaults to ''.
        lesion_suffix (str, optional): [description]. Defaults to '_fracture_refine.nii.gz'.
        instance_suffix (str, optional): [description]. Defaults to '_image_seg_ribs_instance.nii.gz'.
        region_suffix (str, optional): [description]. Defaults to '_image_seg_rib_types.nii.gz'.

     # /mnt/d4/dejuns/ribfrac/infer4refine/anndate_1105/cv04_test_257/man_ai_mask_fpr
    roi_info = {'instance' : ix + 1, 'bbox': roi_bbox, 
                    'class': -1 if roi_cls == 66535 else roi_cls , 
                    'center' : roi_center, 'prob': float(roi_prob)}
    """

    # pid2fns = {a.split(lesion_suffix)[0] : a for a in os.listdir(lesion_dir) if a.endswith(lesion_suffix)}
    
    prop_info_all = []
    for pid, key2fp in pid2niifp_map.items():
        lesion_fp = key2fp['label']
        if lesion_fp is None or not osp.exists(lesion_fp):
            print('!!Exception for ', pid)
            continue
        lesion_fp = Path(lesion_fp)
        roi_json_fp = lesion_fp.parent/f'{pid}_ins2cls.json'
        if roi_json_fp.exists():
            roi_det_infos = load2json(roi_json_fp)
        else:
            lesion_mask_xyz, af_mat = IO4Nii.read(lesion_fp, verbose=False,  dtype = np.uint8) # xyz
            roi_det_infos = roi_info_mask(lesion_mask_xyz)
            save2json(roi_det_infos, lesion_fp.parent, f'{pid}_ins2cls.json')
            
        det_frac_nx8 = np.array([ roi['bbox'] + [roi['prob']] + [roi['class']] for roi in roi_det_infos])
        

        ribinst_fp = structure_dir/f'{pid}{instance_suffix}'
        region_fp = structure_dir/f'{pid}{region_suffix}'
        
        ribinst_mask_zyx, _ = IO4Nii.read(ribinst_fp, verbose= False, dtype = np.uint8, axis_order='zyx')
        region_mask_zyx, af_mat = IO4Nii.read(region_fp, verbose= False, dtype = np.uint8, axis_order='zyx')

        frac_proposal_info, det_ix2cls = get_fracture_info(det_frac_nx8, ribinst_mask_zyx, region_mask_zyx, 
                                                            affine_matrix_raw=af_mat)
        for prop in frac_proposal_info: 
            prop['pid'] = pid
            prop['cohort'] = cohort_id
        
        save2json(frac_proposal_info, lesion_fp.parent, f'{pid}_info_ribs.json')
        # print(f'[Worker{prcs_ix}] pid {pid} proposals {len(frac_proposal_info)}')
        prop_info_all.extend(frac_proposal_info)
    
    return prop_info_all
    
    

if __name__ == '__main__':

    """
    expected directory structure:
    ├── dataset_root
    │   ├── image
    │   │    ├── 1378226_20190624_image.nii.gz
    │   ├── mask_gt
    │   │    ├── 1378226_20190624_fracture.nii.gz
    │   ├── mask_pred
    │   │    ├── 1378226_20190624_image_seg_ribfrac.nii.gz

    │   ├── mask_merge
    │   │    ├── 1378226_20190624_fracture_refine.nii.gz

    """

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir',
                        help="data dir where raw image are stored in nii format ",
                        default = None, 
                        type=str
                        )

    parser.add_argument('--gt_dir',
                        help="data dir where gt mask are stored in nii format ",
                        required=True,
                        type=str
                        )
    parser.add_argument('--pred_dir',
                        help="data dir where pred mask are stored in nii format ",
                        required=True,
                        type=str
                        )
    parser.add_argument('--merge_dir',
                        help="data dir where merge mask will be stored in nii format ",
                        required=True,
                        default = 'man_ai_mask', 
                        type=str
                        )
    parser.add_argument('--label_suffix',
                        help="suffix appending the pid in the filename of gt mask, e.g. 1378226_20190624_fracture.nii.gz",
                        default = '_fracture.nii.gz', 
                        type=str
                        )
    parser.add_argument('--pred_suffix',
                        help="suffix appending the pid in the filename of pred mask, e.g. 1378226_20190624_image_seg_ribfrac.nii.gz",
                        default = '_image_seg_ribfrac.nii.gz', 
                        type=str
                        )
    parser.add_argument('--num_class',
                        help='number of valid class ',
                        default=4,
                        type=int
                        )
    parser.add_argument('--run_pids',
                        help='specify a list pids to run the mask merging',
                        default=[],
                        type=list
                        )
    parser.add_argument('--num_workers',
                        help='number of workers used to run the merging',
                        default=4,
                        type=int
                        )
    parser.add_argument('--maintain_tp',
                        action='store_true',
                        help='keep the label value of tp and not reassign them to 6',
                        )

    args = parser.parse_args()

    # merge scheme: 
    #   FN: (1, 2, 3, 4) 
    #   FP: 5
    #   TP: 6
    num_class, is_maintain_tp = args.num_class, args.maintain_tp # is_maintain_tp: if keep the label value of tp or reassign to 6

    image_dir = args.image_dir
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    merge_dir = args.merge_dir
    label_suffix = args.label_suffix
    pred_suffix = args.pred_suffix
    run_pids = args.run_pids
    num_workers = args.num_workers

    mkdir(merge_dir)
    pid2niifp_map_all = load_list_scan(data_rt=image_dir,  label_rt = gt_dir, extract_roi_info=False, 
                                        label_suffix=label_suffix)
    pids = sorted(list(pid2niifp_map_all)) #[:4]
    print(f'Data {image_dir} \n gt {gt_dir} \n pred {pred_dir} \n merge {merge_dir}')
    print(f'Found {len(pids)} pids')
    if len(run_pids) > 0:
        pids = run_pids
        print('Only run merging for', pids)


    pid2niifp_map = {k:pid2niifp_map_all[k] for k in pids}
    run_parralel(fuse_pred_anno_mask, pid2niifp_map, 
                pred_dir,  merge_dir,  
                label_suffix = label_suffix, 
                pred_suffix = pred_suffix, 
                num_class = num_class, 
                maintain_tp = is_maintain_tp, 
                num_workers = num_workers, 
                )

    # python3 machine_man_refine_standalone.py \
    # --image_dir path/to/image \
    # --gt_dir path/to/gt_mask \
    # --pred_dir path/to/pred_mask \
    # --merge_dir path/to/merge_mask \
    # --label_suffix '_fracture.nii.gz' --pred_suffix '_image_seg_ribfrac.nii.gz'
