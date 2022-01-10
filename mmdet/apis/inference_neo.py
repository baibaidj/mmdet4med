import torch, copy

from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet.core import bbox2result3d, batched_nms_3d
import ipdb

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

def masked_image_modeling(model, img, affine = None, rescale = True,
                          guide_mask = None):

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
    # outer most : tta ;  2nd outer sample/minibatch; inner meta for 1 image
    data_dict.pop('seg_meta_dict', None)
    data_dict.pop('seg', None)
    data_dict['img_metas'] = [{'img_meta_dict': data_dict.pop('img_meta_dict', None)}]
    for k in data_dict.keys(): data_dict[k] = [data_dict[k]] 

    # forward the model
    with torch.no_grad():
        # see models/segmentors/base.py 108, forward method
        # ipdb.set_trace()
        reconstr_images, rand_masks = model(return_loss=False, 
                                        rescale=rescale, 
                                        **data_dict)
        
    return reconstr_images, rand_masks


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