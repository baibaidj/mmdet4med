
import pandas as pd
import os, sys, argparse
mmseg_rt = '/home/dejuns/git/mmseg4med'
monai_rt = '/home/dejuns/git/MONAI'
if mmseg_rt in sys.path: sys.path.remove(mmseg_rt)
if monai_rt in sys.path: sys.path.remove(monai_rt)

import SimpleITK, json
from evaluate_det_result import roi_info_mask
from mmdet.apis.inference_med import init_detector, torch
from mmdet.apis.inference_neo import masked_image_modeling
from mmdet.datasets.transform4med.io4med import *
import cc3d, ipdb
from mmcv import Timer
from mmcv.runner import wrap_fp16_model

def main(cfg, 
        nii_save_dir = '',  
        pid2niifp_map = dict(),
        is_test = False,
        target_spacing = (1.6, 1.6, 1.6),
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

    print('config', config_file)
    print('checkpoint', checkpoint_file)
    # build the model from a config file and a checkpoint file
    # with Timer(print_tmpl='Loading model %s takes {:.3f} seconds' %(cfg.model_name)):
    model = init_detector(str(config_file), str(checkpoint_file), device='cuda:%d' %cfg.gpu_ix)
    if cfg.fp16: 
        wrap_fp16_model(model)
        print('\nFP16 inference')
    
    norm_intense_cfg = dict(type='NormalizeIntensityGPUd',
                                    keys='img',
                                    subtrahend=model.cfg.norm_param['mean'],
                                    divisor=model.cfg.norm_param['std'],
                                    percentile_99_5=model.cfg.norm_param['percentile_99_5'],
                                    percentile_00_5=model.cfg.norm_param['percentile_00_5'])
    # print('\nCases to infer : %d \n' %len(pid2niifp_map))
    pcount = 0
    # infer by model
    for cid, cid_infos in pid2niifp_map.items(): #[1:2]
        img_nii_fp = cid_infos['image']
        # if cid != '1112514_20180429': continue  #'1214845_20181014'  #3837016_image.nii.gz
        print(f'\n[INFER] {pcount} {img_nii_fp}')
        if is_test: continue
        # store_dir
        if cfg.fp16: cid_infos['cid'] = f'{cid}_fp16'
        cid_true = cid_infos['cid']
        reconst_img_fp = osp.join(nii_save_dir, f'{cid_true}_reconstruct.nii.gz')
        if osp.exists(reconst_img_fp): 
            print(f'{reconst_img_fp} exist, so skip inference')
            continue

        img_3d_origin, affine_origin = IO4Nii.read(img_nii_fp, axis_order = None, 
                                                verbose = False, dtype=np.int16)
        # img_ori_size = img_3d_origin.shape
        # # print_tensor(f'Pid {pid} img', img_3d_origin)
        # old_spacing = [abs(affine_origin[i, i]) for i in range(3)]
        # new_shape_raw = [int(img_ori_size[i] * old_spacing[i] / target_spacing[i]) for i in range(3)]

        # # 1. resize to target spacing
        # img_ori_resize = respacing_volume(img_3d_origin, new_shape_raw)
        # img_shrink_size = img_ori_resize.shape
        # affine2model = affine_matrix_i.copy()
        # for i in range(3): affine2model[i, i]  = np.sign(affine_matrix_i[i, i]) * target_spacing[i]

        # body_center = [img_shrink_size[i]//2  for i in range(3)]
        # center_cropper = SpatialCropDJ(body_center, model.cfg.patch_size)
        # img_3d_body, slices4img, slices4patch = center_cropper(img_3d_origin)

        timer = Timer() 
        # 400-16 = 192*2 = 384
        # crop_slicer = tuple([slice(64, 448), slice(64 , 448), slice(None, None)])
        # img_3d_crop = img_3d[crop_slicer]
        reconstr_images, rand_masks = masked_image_modeling(model, img_3d_origin,  
                                        affine = affine_origin, rescale = True, 
                                        target_spacing = target_spacing, 
                                        input_patch_size = model.cfg.patch_size, 
                                        norm_intense_cfg = norm_intense_cfg)
        if cfg.verbose: 
            print_tensor('\t reconstruct images', reconstr_images) 
            print_tensor('\t rand masks', rand_masks)
        reconstr_images = reconstr_images[0, 0].cpu().numpy().astype(np.float32)
        rand_masks = rand_masks[0, 0].cpu().numpy().astype(np.uint8)                                           
        torch.cuda.empty_cache()
        duration  = timer.since_start()
        print(f'\t infer takes {duration:04f}')

        # reconstr_img_resize = np.zeros_like(img_ori_resize, dtype = np.float32)
        # reconstr_img_resize[slices4img] = reconstr_images[slices4patch]
        # rand_mask_resize = np.zeros_like(img_ori_resize, dtype = np.uint8)
        # rand_mask_resize[slices4img] = rand_masks[slices4patch]
        # reconstr_img_origin = respacing_volume(reconstr_img_resize, img_ori_size)
        # rand_mask_origin = respacing_volume(rand_mask_resize, img_ori_size)
        
        IO4Nii.write(reconstr_images, nii_save_dir, f'{cid_true}_reconstruct', affine_origin)
        IO4Nii.write(rand_masks, nii_save_dir, f'{cid_true}_rand_masks', affine_origin)
        pcount += 1

    # per_case_fp = osp.join(nii_save_dir, f'aug_inference_{fold_ix_str}.csv')
    # result_tb = pd.DataFrame(result_by_case)
    # result_tb.to_csv(per_case_fp, index = False)
    # print(result_tb.describe())
    return #per_case_fp


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
        pid2niifp_map = load_list_detdj(cfg.data_rt, mode = cfg.split, json_filename = 'dataset_1231.json')
    
    pid2niifp_map = sample_list2run(pid2niifp_map, run_pids=cfg.run_pids, 
                                            num_fold=cfg.num_fold, 
                                            fold_ix=cfg.fold_ix)
    fold_ix_str = f'{cfg.fold_ix}@{cfg.num_fold}'

    git_rt = Path(cfg.repo_rt)
    model_store_dir = git_rt/cfg.model_name
    pos_thresh_str = str(cfg.pos_thresh).replace('.', '-')
    store_name =  str(f'visual_{cfg.dataset_name}_{cfg.split}_nii')
    nii_save_dir = model_store_dir/store_name
    mkdir(nii_save_dir)

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