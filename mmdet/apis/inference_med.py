# volume based inference for medical images like CT
from mmdet.apis.inference import *
from mmdet.datasets.transform4med.io4med import *
from mmdet.core.evaluation.mean_dice import *
from tqdm import tqdm
from mmcv import Timer
import cc3d, copy


class LoadImageMonai:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_meta_dict'] = dict(original_affine = results['affine'], 
                                        affine = results.pop('affine', None),
                                        spatial_shape = img.shape,
                                        filename_or_obj = results.get('filename', ''), 
                                        guide_mask = results.pop('guide_mask', None) )
        return results


def inference_detector4med(model, img, affine = None, rescale = True,
                           need_probs = True, guide_mask = None, 
                            target_spacings = None):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        result_prob: post softmax/sigmoid before binarization
        feature_map: 
        data: 

    """
    # assert isinstance(need_feature_index, (type(None), int))
    cfg = copy.deepcopy(model.cfg)
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # NOTE: make the target spacings adpative to the affine_matrix 
    if isinstance(target_spacings, (list, tuple)): # make the target spacings adpative to the affine_matrix 
        cfg.data.test.pipeline[1].target_spacings = target_spacings

    test_pipeline = [LoadImageMonai()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    if not isinstance(img, (tuple, list)):
        img = [img]
    num_samples = len(img)
    data = []
    for img_i in img:
        this_data = dict(img=img_i, affine = affine, guide_mask = guide_mask)
        this_data = test_pipeline(this_data)
        data.append(this_data)
    
    data = collate(data, samples_per_gpu=num_samples)
    # pdb.set_trace()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        # see models/segmentors/base.py 108, forward method
        det_results, seg_results = model(return_loss=False, 
                                        rescale=rescale, 
                                        need_probs = need_probs, 
                                        **data)
    return det_results, seg_results


async def async_inference_detector4med(model, img, affine = None, rescale = True,
                                        need_probs = True, guide_mask = None, 
                                        target_spacings = None):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    # assert isinstance(need_feature_index, (type(None), int))
    cfg = copy.deepcopy(model.cfg)
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # NOTE: make the target spacings adpative to the affine_matrix 
    if isinstance(target_spacings, (list, tuple)): # make the target spacings adpative to the affine_matrix 
        cfg.data.test.pipeline[1].target_spacings = target_spacings

    test_pipeline = [LoadImageMonai()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    if not isinstance(img, (tuple, list)):
        img = [img]
    num_samples = len(img)
    data = []
    for img_i in img:
        this_data = dict(img=img_i, affine = affine, guide_mask = guide_mask)
        this_data = test_pipeline(this_data)
        data.append(this_data)
    
    data = collate(data, samples_per_gpu=num_samples)
    # pdb.set_trace()
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    # see models/segmentors/base.py 108, forward method
    det_results, seg_results = await model.aforward_test(return_loss=False, 
                                                        rescale=rescale, 
                                                        need_probs = need_probs, 
                                                        **data)

    return det_results, seg_results


denormalize = lambda x: 2.0 * x - 1.0
# object_ratio_func = lambda x: x.sum()/np.prod(x.shape) * 100

class Inferencer(object):

    def __init__(self, model, 
                    batch_size = 8, num_classes = 2, 
                    is_3d_input = False, is_3d_output = False, 
                    infer_z_step = None,
                    point4keep = 64, feat_ix = -1,  is_use_entropy = False, 
                    is_add_loc = False,
                    store_dir = None, verbose = False, is_plot = False,
                    CLASSES = None, label_mapping = None, 
                    ) -> None:

        self.model = model
        self.batch_size = batch_size
        self.config = model.cfg
        self.in_slice = model.cfg.get('in_slice', 1)#if hasattr(model.cfg, 'in_slice') else model.cfg.in_channels
        self.num_classes = num_classes
        self.is_3d_input = is_3d_input
        self.is_3d_output = is_3d_output
        self.infer_z_step = infer_z_step if infer_z_step else self.in_slice//2
        self.view_channel = self.config.get('view_channel',  None)
        self.axis_order = view2axis[self.view_channel]

        # self.is_add_z = is_add_z
        self.is_use_entropy = is_use_entropy
        self.is_add_loc = is_add_loc
        self.point4keep = point4keep
        self.feat_ix = feat_ix

        self.store_dir = store_dir
        self.verbose = verbose
        self.is_plot = is_plot
        self.CLASSES = CLASSES
        self.label_mapping = label_mapping

        if self.store_dir is not None:
            mkdir(self.store_dir)

    def init_pid_dir(self, pid = ''):
        pid_dir = None
        if self.store_dir is not None:
            pid_dir = osp.join(self.store_dir, pid)
            mkdir(pid_dir)
        return pid_dir


    def infer1volume_monai(self, img_3d, affine = None, is_rescale = True, 
                           need_logits = False, 
                           fp16 = False):

        # img_draw = ImageDrawer(img_3d, self.in_slice)
        if self.verbose: print_tensor(f'[CTVolume]', img_3d) 
        timer = Timer() 
        # if verbose and mask_i.max() < 1: continue
        det_result, seg_result = inference_detector4med(self.model, img_3d, 
                                                        affine = affine,
                                                        rescale = is_rescale,
                                                        need_probs = need_logits, 
                                                        fp16 = fp16)
        torch.cuda.empty_cache()

        if self.verbose: print_tensor('\tseg logits', seg_result)

        duration  = timer.since_start()
        return det_result, seg_result, duration

    @staticmethod
    def prob2mask(seg_logit : torch.Tensor, class_dim = 1):
        seg_pred = seg_logit.argmax(dim=class_dim) if seg_logit.shape[class_dim] > 1 else seg_logit.squeeze(class_dim) > 0.5
        seg_pred = seg_pred.short().cpu().numpy()
        return seg_pred

    def prob2mask2save(self, pred_4d, mask_gt_3d, affine_matrix, store_dir, img_nii_fp, average_time = 0.0):
        pred_3d_raw = self.prob2mask(pred_4d)
        print_tensor('\tprob-by-value', pred_3d_raw)
        with Timer(print_tmpl='\tcompute metric {:.3f} seconds'):
            pred_lysis_raw, dice_raw = self.measure_pred(pred_3d_raw, mask_gt_3d)
        dice_fine = 0.0
        basename = '%s_%0.4f_%0.4f' %( str(img_nii_fp).split(os.sep)[-4] , dice_raw, dice_fine)
        print('\tsave to ', basename)
        if self.store_dir is not None:
            IO4Nii.write(pred_lysis_raw, store_dir, basename + 'raw', affine_matrix, 
                         axis_order= self.axis_order )
            # IO4Nii.write(pred_lysis_fine, self.store_dir, basename, affine_matrix)
        return dice_raw, dice_fine

    def seg_prob2save(self, pred_4d, affine_matrix, store_dir = None, filename = None, 
                            average_time = 0.0, mask_gt_3d = None, 
                            class_dim = 1, metric_keys = ('dice',),
                            axis_order = None, label_mapping4pred = None):
        store_path = None
        if len(pred_4d.shape) >= 4: 
            pred_3d_raw = self.prob2mask(pred_4d, class_dim = class_dim)
        else: pred_3d_raw = pred_4d.numpy() if isinstance(pred_4d, torch.Tensor) else pred_4d
        
        if mask_gt_3d is not None: 
            if mask_gt_3d.shape != pred_3d_raw.shape:
                Warning(f'pred shape {pred_3d_raw.shape} not identical to gt shape {mask_gt_3d.shape}')
                mask_gt_3d = None
        if self.verbose: print_tensor('\tprob-by-value', pred_3d_raw)
        # with Timer(print_tmpl='\tcompute metric {:.3f} seconds'):
        metric2results = evaluate_pred_against_label(pred_3d_raw, mask_gt_3d, metric_keys, self.num_classes)
        if self.verbose: print(f'\tMetricResult: ', metric2results)
        metric_str_classes = []
        for i, cls in enumerate(self.CLASSES[1:]):
            metric_list = tuple([v[i] for k, v in metric2results.items()])
            metric_str_format = '|'.join(['%s'] + ['%0.3f'] * len(metric_keys))
            # print('Check metric str format', metric_str_format)
            cls_metric_str = metric_str_format %(cls, *metric_list)
            metric_str_classes.append(cls_metric_str)
        score_by_class = '_'.join(metric_str_classes)
        # score_by_class = '_'.join(['%s%0.2f'%(cls, dice3d_list[i] * 100) for i, cls in enumerate(self.CLASSES) if i != 0])
        basename = '%s_%s_%0.2fs' %(filename, score_by_class, average_time)
        # pdb.set_trace()
        self.verbose: print(f'\tsave to {basename} axis order {self.axis_order}')
        if label_mapping4pred is not None: pred_3d_raw = convert_label(pred_3d_raw, label_mapping4pred)
        if store_dir is not None:
            if axis_order is None: axis_order = self.axis_order 
            store_path = IO4Nii.write(pred_3d_raw, store_dir, basename, affine_matrix, 
                            axis_order= axis_order)

            # IO4Nii.write(pred_lysis_fine, self.store_dir, basename, affine_matrix)
        return pred_3d_raw, store_path, metric2results

    def seg_reg2save(self, pred_reg_4d, affine_matrix, store_dir, filename, average_time = 0.0):
        pred_3d_raw = pred_reg_4d.squeeze().cpu().numpy()#prob2mask(pred_4d)
        pred_3d_raw = pred_3d_raw.astype(np.float32)
        print_tensor('\tprob-by-value', pred_3d_raw)

        basename = '%s_%0.4fs_sdm' %(filename, average_time)
        print('\tsave to ', basename)
        if self.store_dir is not None:
            store_path = IO4Nii.write(pred_3d_raw, store_dir, basename + 'raw', affine_matrix, 
                         axis_order= self.axis_order)
        else:
            store_path = None
            # IO4Nii.write(pred_lysis_fine, self.store_dir, basename, affine_matrix)
        return store_path


    def infer1volume(self, img_3d, caseid = '', store_dir = None, affine_matrix = None,
                     mask_gt_3d = None, slice_per_chunk = 256):
        """
        Args:
            img_3d: dimension order ZYX
            caseid: 
            store_dir: if given, prediction will be saved to this dir
            affine_matrix: the affine matrix of the CT series
            mask_3d: same dimension order as img_3d ZYX. if given, metrics will be computed and return

        return:
            result_dict:
                {   'caseid': 
                    'infer_time_avg': inference time for one case
                    'pred_mask': segmentation mask in shape ZYX of lung, heart, liver and kideny 
                    'img_dim{#}' : image shape of dimension #
                    'spacing_dim{#}' : image spacing of dimension #
                    'cls_metric' : metric value against gt if given
                }
        """
        this_result = {'caseid' : caseid}
        this_result.update({f'img_dim{i}': img_3d.shape[i] for i in range(3)})
        if self.verbose: print_tensor(f'Infer caseid {caseid}, image info:', img_3d)
        if affine_matrix is not None: 
            this_result.update({f'spacing_dim{i}': affine_matrix[i, i] for i in range(3)})
        num_slice = img_3d.shape[0]
        num_chunk, last_chunk = int(num_slice//slice_per_chunk), int(num_slice % slice_per_chunk)
        if last_chunk > slice_per_chunk/2: num_chunk += 1
        num_chunk = max(1, num_chunk)
        pred_in_chunks = []
        average_time = 0
        for cix in range(num_chunk):
            start, end  = cix * slice_per_chunk, (cix + 1)* slice_per_chunk
            if cix == num_chunk - 1: end = num_slice
            img_3d_chunk = img_3d[start:end]
            # print_tensor(f'{cix} {start}:{end} this chunk ', img_3d_chunk)
            pred_3d, avg_time, *_ = self.infer1volume2d(img_3d_chunk, is_rescale = True, 
                                                        batch_size = self.batch_size) # 32 > 6G
            pred_in_chunks.append(pred_3d)
            average_time += avg_time

        pred_3d_final = torch.cat(pred_in_chunks, dim = 0)
        if self.verbose:  print_tensor(f'\tPrediction taking {average_time}', pred_3d_final)
        pred_mask_3d, _, metric2results = self.seg_prob2save(pred_3d_final, affine_matrix, store_dir, 
                                                        filename = caseid, average_time = average_time, 
                                                        mask_gt_3d= mask_gt_3d, metric_keys = ('dice',),
                                                        class_dim = 1, axis_order = 'zyx')  
        torch.cuda.empty_cache()                                              
        this_result[f'infer_time_avg'] = average_time

        if metric2results is not None:
            for metric_key, result_by_class in metric2results.items():
                for i, result in enumerate(result_by_class): 
                    this_result[f'{self.CLASSES[i + 1]}_{metric_key}'] = result
        this_result['pred_mask'] = pred_mask_3d
        return this_result



def reserve_center_crop(seg_logit, original_shape):
    assert len(original_shape) == 2
    h_pre, w_pre = seg_logit.shape[-2:]   
    h_resize, w_resize = original_shape
    # print('\n\tTest: ip2model %s, pre center crop %s' %((h_pre, w_pre), (h_resize, w_resize)))
    if (h_pre, w_pre) != (h_resize, w_resize):
        pred_container = torch.zeros(seg_logit.shape[:-2] + (h_resize, w_resize), 
                                    dtype=seg_logit.dtype, device = seg_logit.device) + seg_logit.min()
        if seg_logit.shape[1] > 1: pred_container[:, 0, ...] = seg_logit.max() # 0th channel is background and should be largest  
        # print_tensor('\t    raw pred', seg_logit)
        # print_tensor('\t    pred_container', pred_container)
        h_tic, w_tic = (h_resize - h_pre)//2, (w_resize - w_pre)//2
        pred_container[..., h_tic: h_tic+ h_pre, w_tic: w_tic + w_pre ] = seg_logit
        seg_logit = pred_container
    return seg_logit

def max_region_binary(mask_binary):
    # centerline_out_cc = np.zeros(centerline_out.shape, dtype=np.uint8)
    # centerline_out_cc[centerline_out > 0] = 1
    # centerline_out_cc = cc3d.connected_components(centerline_out_cc)
    # centerline_out_cc_max = np.argmax(np.bincount(centerline_out_cc[centerline_out_cc > 0].flatten()))
    # centerline_out[centerline_out_cc != centerline_out_cc_max] = 0
    mask_regions = cc3d.connected_components(mask_binary)
    mask_max_index = np.argmax(np.bincount(mask_regions[mask_regions > 0].flatten()))
    mask_binary[mask_regions != mask_max_index] = 0
    return mask_binary


def grey_max_region(mask_multi_value, target_values = (1,2,3)):
    if target_values is None: 
        target_values = np.unique(mask_multi_value)[1:]
    
    mask_result = np.zeros_like(mask_multi_value)
    for tv in target_values:
        this_mask_max = max_region_binary(mask_multi_value == tv)
        mask_result[this_mask_max.astype(bool)] = tv
    return mask_result


from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet.core import bbox2result3d, batched_nms_3d

def tta_detect_1by1(model, img, affine = None, rescale = True,
                            need_probs = True, guide_mask = None, target_spacings = [None]):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        result_prob: post softmax/sigmoid before binarization
        feature_map: 
        data: 

    """

    cfg = copy.deepcopy(model.cfg)
    device = next(model.parameters()).device  # model device
    # 0. ToTensor ToGPU add channel
    data_dict = LoadImageGPU()(dict(img=img, affine = affine, seg = guide_mask), device = device)
    normalizer = build_from_cfg(dict(type='NormalizeIntensityGPUd',
                                    keys='img',
                                    subtrahend=330,
                                    divisor=562.5,
                                    percentile_99_5=3071,
                                    percentile_00_5=-927), PIPELINES)
    # pdb.set_trace()
    # 1. normalize image
    # collect_keymap = {'img' : 'img', 'seg': 'seg', 'img_metas': 'img_meta_dict'}
    data_dict = normalizer(data_dict)
    det_results_tta = []
    # NOTE: Try two spacing, rather than two view
    for new_spacing in target_spacings:
        for flip_direction in [None, ]: #, 'diagonal'
            # if new_spacing is None and flip_direction is not None:
            #     continue
            print(f'\n[DetTTA] new spacing {new_spacing}  flip {flip_direction}')
            resizer = ResizeTensor5DGPU(keys = ('img', 'seg'), new_spacing = new_spacing)
            flipper = FlipTensor5DGPU(keys = ('img', 'seg'), flip_direction = flip_direction)
            # 1. respacing
            data_var = resizer(**data_dict)
            # 2. FlipTTA
            data_var = flipper(**data_var)
            # outer most : tta ;  2nd outer sample/minibatch; inner meta for 1 image
            data_var.pop('seg_meta_dict', None)
            data_var['img_metas'] = [{'img_meta_dict': data_var.pop('img_meta_dict', None)}]
            for k in data_var.keys(): data_var[k] = [data_var[k]]

            # forward the model
            with torch.no_grad():
                # see models/segmentors/base.py 108, forward method
                det_results, *seg_results = model(return_loss=False, 
                                                rescale=rescale, 
                                                need_probs = need_probs, 
                                                **data_var)
            det_results_tta.extend(det_results)
            # del data_var; torch.cuda.empty_cache()
    # detection
    bbox_nx7 = torch.cat([d1[0] for d1 in det_results_tta], axis = 0) # nx7
    label_nx1 = torch.cat([d1[1] for d1 in det_results_tta], axis = 0) # nx1

    if bbox_nx7.shape[0] < 1: 
    # pdb.set_trace()
        return [[torch.zeros((0, 7)), ] ], None
        
    dets_bbox_nx7, keep_idx = batched_nms_3d(bbox_nx7[:, :6], bbox_nx7[:, 6], 
                                            label_nx1, cfg.model.test_cfg['nms'])
    label_nx1 = label_nx1[keep_idx]
    max_per_img, num_classes = cfg.model.test_cfg['max_per_img'], cfg.num_classes
    if max_per_img > 0:
        dets_bbox_nx7 = dets_bbox_nx7[: max_per_img]
        label_nx1 = label_nx1[:max_per_img]
    # pdb.set_trace()
    bbox_results = [bbox2result3d(dets_bbox_nx7, label_nx1, num_classes)]
    return bbox_results, seg_results



class LoadImageGPU:
    """A simple pipeline to load image."""

    def __call__(self, results, device = 'cuda:0'):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        results['img_meta_dict'] = dict(original_affine = results['affine'], 
                                        affine = results.pop('affine', None),
                                        spatial_shape = results['img'].shape ,
                                        filename_or_obj = results.get('filename', ''), 
                                        # new_pixdim = None, 
                                        # flip = False, 
                                        # flip_direction = 'diagnal'
                                        )
        results['seg_meta_dict'] = dict(**results['img_meta_dict'])

        results['img'] = torch.from_numpy(results['img']).float().to(device)[None, None]
        if results['seg'] is not None:
            results['seg'] = torch.from_numpy(results['seg']).float().to(device)[None, None]
        return results


import torch.nn.functional as F
class ResizeTensor5DGPU(object):
    """
    Resize 5D tensor on GPU
    """

    def __init__(
        self,
        keys,
        new_spacing ,
        mode = ['trilinear', 'nearest'],
        align_corners = [None, None],
        dtype = [torch.float, torch.long],
        meta_key_postfix: str = "meta_dict",
        verbose = False,

    ) -> None:
        """
        Args:

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__()
        self.keys = keys
        self.new_spacing = new_spacing
        self.mode = mode
        self.align_corners = align_corners #, len(self.keys))
        # self.dtype = dtype # ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix
        self.verbose = verbose

    def __call__(self, **data):
        d = dict(data)
        # new_pixdim = d['img_meta_dict'].get('new_pixdim', None)
        # print('Check new pixdim', new_pixdim)
        if self.new_spacing is None: return d    

        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            old_shape = d[key].shape[2:]
            old_spacing = [abs(meta_data['affine'][i, i]) for i in range(3)]
            new_shape = [round(s * old_spacing[i] / self.new_spacing[i]) for i, s in enumerate(old_shape)]

            d[key] = F.interpolate(d[key], size = new_shape, 
                                    mode=self.mode[idx], 
                                    align_corners=self.align_corners[idx])
            if self.verbose: print(f'\tRespacing: from {old_shape} {old_spacing} to {new_shape} {self.new_spacing}')
            # set the 'affine' key
            # meta_data["affine"] = self.new_spacing #TODO: get it right
            meta_data['shape_post_resize'] = new_shape
        return d


class FlipTensor5DGPU(object):
    """
    
    Flip Last three dimensions which are the spatial ones
    """
    def __init__(
        self,
        keys,
        flip_direction = 'diagnoal',
        meta_suffix = 'meta_dict'
    ) -> None:

        self.keys = keys
        # self.flip = flip
        self.flip_direction = flip_direction
        self.meta_suffix = meta_suffix

    def __call__( self, **data):
        # self.randomize()
        d = dict(data)
        # flip = d['img_meta_dict']['flip']
        # flip_direction = d['img_meta_dict']['flip_direction']

        if self.flip_direction == 'diagonal': flip_dims = (2, 3)
        elif self.flip_direction == 'headfeet': flip_dims = (4, )
        elif self.flip_direction == 'all3axis': flip_dims = (2, 3, 4)
        else: flip_dims = None
        # print(f'PIPELINE: Flip {flip} Direction {flip_direction} dims {flip_dims}')
        if flip_dims is not None: 
            for key in self.keys:
                d[key] = torch.flip(d[key], flip_dims)
                d[f'{key}_{self.meta_suffix}']['flip'] = None
                d[f'{key}_{self.meta_suffix}']['flip_direction'] = self.flip_direction
                
        return d

