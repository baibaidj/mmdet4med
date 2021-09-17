import numpy as np
from .load_nn import load_json, load_pickle, Any
from ..builder import PIPELINES
from .io4med import print_tensor, random, IO4Nii
import sys, pdb

from monai.transforms.croppad.array_ import SpatialCrop_
from monai.utils import fall_back_tuple, ensure_tuple_rep
# from monai.transforms.io.array import LoadImage, SaveImage



@PIPELINES.register_module()
class Load1CaseDet:
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
                'spine_boudnary': list(int), (x, y) NOTE: x0x1y0y1, not x0y0x1y1
                }
    """
    def __init__(self, keys = ('img', 'seg', 'roi'), 
                 meta_key = 'img_meta_dict', verbose = False,
                 semantic2binary = True
                ) -> None:
        self.keys = keys
        self.meta_key = meta_key
        self.verbose = verbose
        self.semantic2binary = semantic2binary
        # self.axis_reorder = axis_reorder

    def __call__(self, data):
        data = dict(data)
        # a = [print('[loadnn]', k, v) for k, v in data.items()]
        data[self.meta_key] = dict(spatial_shape = None, original_affine = None, 
                                    filename_or_obj = None)
        img_fp, affine_matrix = None, None
        for k in self.keys:
            fp = data.get(k + '_fp', None)
            if fp is None:
                data[k] = None
            else:
                if fp.endswith('nii') or fp.endswith('nii.gz'):
                    data[k], affine_matrix = IO4Nii.read(fp, verbose=False, axis_order=None, dtype=np.int16)
                    data[self.meta_key]['spatial_shape'] = data[k].shape
                    img_fp = fp
                    if self.verbose: print_tensor(f'[load] {k}', data[k])
                elif fp.endswith('json'):
                    data[k] = load_json(fp)
                    if self.verbose: print(f'[load] {k}', data[k])
                else:
                    raise ValueError(f'load only nii(.gz) or pkl but got {fp}')
        if 'set' in data:
            if data['seg'].shape != data['img'].shape:
                print('[Warn] mask shape not match image shape', data['seg'].shape, data['img'].shape)
        # data['instance_mapping'] = data['property'].pop('instances')    
        data[self.meta_key]['original_affine'] = affine_matrix
        data[self.meta_key]['filename_or_obj'] = img_fp
        if 'roi' in data.keys():
            data[self.meta_key]['inst2cls_map'] = {roi['instance']: 0 if self.semantic2binary 
                                                    else (roi['class'] - 1) for roi in data['roi']}
        return data

coin_func = lambda : random.random() > 0.5
def center_range_dim(bg_size, crop_size):
    if crop_size >= bg_size:
        return (bg_size//2, bg_size//2+1)
    else:
        return (crop_size//2, bg_size - crop_size//2)
def sample_from_range(tic, toc):
    if tic == toc:  return tic
    if tic > toc:
        tic, toc = toc, tic
    return random.randrange(tic, toc)

    

@PIPELINES.register_module()   
class InstanceBasedCropDet:
    """
    rois: list[dict(), dict(), ...]
    one_dict: {'instance' : int, start from 1
                'bbox': list(int), e.g. (x1, y1, z1, x2, y2, z2)
                'class': int, start from 1 
                'center' : list(int), (x, y, z)
                'spine_boudnary': list(int), (x, y) NOTE: x0x1y0y1, not x0y0x1y1
                }
    """
    def __init__(self, 
                keys, 
                patch_size, 
                oversample_classes = (1, 2), 
                img_key = None,
                verbose = False,
                num_sample = 3, 
                ) -> None:
        self.keys = keys
        self.oversample_classes = oversample_classes
        self.img_key = keys[0] if img_key is None else img_key
        self.patch_size = patch_size
        # self.pyramid_scale = pyramid_scale
        self.verbose = verbose
        self.num_sample = num_sample
        self.instance_key = 'instance_ix'

    def __call__(self, data) -> Any:

        data = dict(data)
        self.cid = data['cid']
        image_shape = data[self.img_key].shape[-3:]
        self.patch_size = fall_back_tuple(self.patch_size, default=image_shape)
        instance_ix = data.get(self.instance_key, -1)
        sample_centers = self.create2center_per_instance(data.get('roi', []), instance_ix, image_shape)

        results = []
        for c3 in sample_centers:
            # create cropper    
            cropper =  SpatialCrop_(roi_center=tuple(c3), roi_size=self.patch_size)
            crop_data = {k: data[k] for k in data.keys() if k not in self.keys}
            # crop key by key
            for k in self.keys:
                prior_shape = data[k].shape
                crop_data[k] = cropper(data[k])
                crop_data['img_meta_dict'][f'cropshape'] = tuple(crop_data[k].shape[1:])
                if self.verbose: print_tensor(f'[InsCrop] {self.cid} {k} {c3} pre {prior_shape} after', crop_data[k])
            results.append(crop_data)
        return results

    def create2center_per_instance(self, case_rois, instance_ix, image_shape):
        roi_info = [r for r in case_rois if r.get('instance', 0) == instance_ix]
        roi_info = roi_info[0] if len(roi_info) > 0 else None

        num_pos_sample = 1 if roi_info is not None else 0
        num_neg_sample = 2 - num_pos_sample
        sample_centers = []
        if num_pos_sample > 0:
            pos_center = self.random_foreground_center(roi_info['bbox'])
            sample_centers.append(pos_center)
        for _ in range(num_neg_sample):
            neg_center_c3 = self.rand_rib_region(image_shape, self.patch_size)
            sample_centers.append(neg_center_c3)
            if self.verbose: print(f'\t{self.cid} NEG center {neg_center_c3}')
        return sample_centers

    def rand_rib_region(self, image_shape, patch_shape, verbose = False):
        center_bg = []
        x_range, y_range, z_range = [center_range_dim(image_shape[i], patch_shape[i]) for i in range(3)]
        pick_in_4 = random.randrange(0, 4)
        if pick_in_4 == 0:
            # fix x_min, float y 
            cx = x_range[0]
            cy = random.randrange(y_range[0], y_range[1])
        elif pick_in_4 == 1:
            # fix x_max, float y
            cx = x_range[1]
            cy = random.randrange(y_range[0], y_range[1])
        elif pick_in_4 == 2:
            # fix y_min, float x
            cx = random.randrange(x_range[0], x_range[1]) 
            cy = y_range[0]
        else:
            # fix y_max, float x
            cx = random.randrange(x_range[0], x_range[1]) 
            cy = y_range[1]
        cz = random.randrange(z_range[0], z_range[1])
        center_bg = (cx, cy, cz)
        if self.verbose: print(f'[InstCrop] rand rib region {pick_in_4} c3 {center_bg} ')
        return center_bg


    def create3centers_per_case(self, case_rois, image_shape):
        num_rois = len(case_rois)
        fg_ixs = []
        if num_rois > 0:
            rand_fg_ix = sample_from_range(0, num_rois)
            fg_ixs.append(rand_fg_ix)
            if isinstance(self.oversample_classes, (tuple, list)):
                os_ixs = [i for i in range(num_rois) if case_rois[i]['class'] in self.oversample_classes]
                if len(os_ixs) != 0 and len(os_ixs) != num_rois:
                    for _ in range(3):
                        rand_os_ix = random.choice(os_ixs)
                        if rand_os_ix != rand_fg_ix:
                            fg_ixs.append(rand_os_ix)
                            break
        # print(f'\n[InsCrop] {self.cid} {roi_info}')
        # get instance bbox
        # sample center from foreground
        sample_centers = []
        if len(fg_ixs) == 0:
            sample_centers = [self.rand_rib_region(image_shape, self.patch_size) for i in range(self.num_sample)]
        else:
            is2fg = coin_func()
            num_pos_sample = min(2, len(fg_ixs)) if is2fg else 1
            num_neg_sample = self.num_sample - num_pos_sample
            if self.verbose: print(f'\n[InsCrop] {self.cid}  POScenter {num_pos_sample} NEGcenter{num_neg_sample}')
            for p in range(num_pos_sample):
                roi_info = case_rois[fg_ixs[p]]
                fg_ix_center = self.random_foreground_center(roi_info['bbox'], verbose=self.verbose)
                sample_centers.append(fg_ix_center)
                if self.verbose: print(f'\t{self.cid} POS center {fg_ix_center} inst {roi_info} imgshape {image_shape} ')
            for _ in range(num_neg_sample):
                neg_center_c3 = self.rand_rib_region(image_shape, self.patch_size)
                sample_centers.append(neg_center_c3)
                if self.verbose: print(f'\t{self.cid} NEG center {neg_center_c3}')
        return sample_centers


    def random_foreground_center(self, bbox_c6, is_nn_order = False, 
                                zyx2xyz = False,  verbose = False): # 
        """
        nnDet: dim order zyx, image permute to xyz, here bbox also should be permuted 
               z reordered

        Args:
            bbox_c6 ([type]): [description]
            image_shape ([type]): [description]
            is_nn_order (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        assert len(bbox_c6) == 6
        # bboxes: [x1, y1, x2, y2, z1, z2]

        x1, y1, z1, x2, y2, z2 = convert_coord_nn(bbox_c6, is_nn_order,  zyx2xyz)

        # if isinstance(z_size, int): 
        #     z1, z2 = max(z_size - z2, 0), max(z_size - z1, 0)
        if verbose: print(f'\t[FGCenter] xyz1,xyz2 ', x1, y1, z1, x2, y2, z2)
        center_fg = []
        for center_min, center_max in zip((x1, y1, z1), (x2, y2, z2)):
            rand_center = random.randrange(center_min, center_max)
            center_fg.append(rand_center)
        return center_fg


    def select_background_center(self, image_shape, patch_shape, case_rois):
        """
        
        """
        assert len(image_shape) == len(patch_shape)

        safe_min2max = lambda l : l if l[0] < l[1] else l[::-1]
        spine_bound_xy = np.array(case_rois[0]['spine_boundary']) # 

        w, h, d = image_shape
        # sample spine
        spine_rangexyz = safe_min2max([max(w // 2 - 50, 0), min(w // 2 + 50, w)]) + \
                        safe_min2max([int(a) for a in spine_bound_xy[2:]]) + \
                        safe_min2max([patch_shape[2]//2 , min(d - patch_shape[2]//2, d)])
        spine_rangexyz = np.array(spine_rangexyz).reshape((3, 2))
        # if coin_func(): 
        sample_bg_center = [sample_from_range(bound[0], bound[1]) for i, bound in enumerate(spine_rangexyz)]
        # else:
        #     # sample opposite of an roi
        #     rand_roi_ix = random.randrange(0, len(case_rois))
        #     rand_roi_c3 = case_rois[rand_roi_ix]['center']
        #     c3_opposite = [a for a in rand_roi_c3]
        #     for dim, cix in enumerate(rand_roi_c3):
        #         if dim == 0: c3_opposite[dim] = w - cix if coin_func() else cix
        #         elif dim == 1: c3_opposite[dim] = cix
        #         else: 
        #             bound = spine_rangexyz[2]
        #             c3_opposite[dim] = sample_from_range(bound[0], bound[1])
        #     self.verbose: print(f'[InstCrop] opposite {c3_opposite} of foreground  {rand_roi_c3}')
        #     sample_bg_center = c3_opposite
        return sample_bg_center

@PIPELINES.register_module()
class RemoveRoiByClass:

    def __init__(self, exclude_classes = (-1, ), verbose = False) -> None:
        self.exclude_classes = (exclude_classes, ) if isinstance(exclude_classes, int) else exclude_classes
        self.verbose = verbose
    def __call__(self, data) -> Any:
        data = dict(data)
        self.cid = data['cid']

        for roi in data['roi']:
            roi_cls = roi['class']
            if roi_cls in self.exclude_classes:
                roi_inst = roi['instance']
                if self.verbose: print(f'[MaskClean] pid {self.cid} cls {roi_cls} inst {roi_inst} removed ')
                roi_bbox = roi['bbox']
                roi_slicer = tuple([slice(roi_bbox[i], roi_bbox[i+3]) for i in range(3)])
                data['seg'][roi_slicer] = 0
        return data

@PIPELINES.register_module()
class Load1CaseNN:
    """
    property:
    # original_size_of_raw_data :	 [380 512 512]
    # original_spacing :	 [1.         0.79296875 0.79296875]
    # list_of_data_files :	 [PosixPath('/data/lung_algorithm/data/nnDet_data/Task020FG_RibFrac/raw_splitted/imagesTr/RibFrac343_0000.nii.gz')]
    # seg_file :	 /data/lung_algorithm/data/nnDet_data/Task020FG_RibFrac/raw_splitted/labelsTr/RibFrac343.nii.gz
    # itk_origin :	 (-226.603515625, -389.103515625, -822.7999877929688)
    # itk_spacing :	 (0.79296875, 0.79296875, 1.0)
    # itk_direction :	 (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # instances :	 {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
    # crop_bbox :	 [(0, 380), (0, 512), (0, 512)]
    # classes :	 [-1.  0.  1.  2.  3.  4.  5.]
    # size_after_cropping :	 (380, 512, 512)
    # size_after_resampling :	 (304, 550, 550)
    # spacing_after_resampling :	 [1.25       0.73828101 0.73828101]
    # use_nonzero_mask_for_norm :	 OrderedDict([(0, False)])


    bboxes: # zyx
    # boxes: [[ 23 314  33 341  76  91] 
    #         [166 164 185 211 377 414]
    #         [141 159 162 225 388 442]
    #         [117 175 138 234 427 462]
    #         [101 201 121 266 451 478]]
    # instances :	[1, 2, 3, 4, 5]
    # labels :	 [0, 0, 0, 0, 0]
    
    """
    def __init__(self, np_load_mode = 'r', 
                 keys = ('img', 'seg', 'property', 'bboxes'), 
                 meta_key = 'img_meta_dict', verbose = False,
                 axis_reorder = True
                ) -> None:
        self.memmap_mode = np_load_mode
        self.keys = keys
        self.meta_key = meta_key
        self.verbose = verbose
        self.axis_reorder = axis_reorder

    def __call__(self, data):
        data = dict(data)
        # a = [print('[loadnn]', k, v) for k, v in data.items()]
        data['img_meta_dict'] = dict(spatial_shape = None, original_affine = None)
        for k in self.keys:
            fp = data.get(k + '_fp', None)
            if fp is None:
                data[k] = None
            else:
                if fp.endswith('npy') or fp.endswith('npz'):
                    data[k] = np.load(fp, allow_pickle=True) #self.memmap_mode,
                    if self.axis_reorder: data[k] = array_zyx2xyz(data[k])
                    data['img_meta_dict']['spatial_shape'] = data[k].shape
                    if self.verbose: print_tensor(f'[load] {k}', data[k])
                elif fp.endswith('pkl'):
                    data[k] = load_pickle(fp)
                    if self.verbose: print(f'[load] {k}', data[k])
                else:
                    raise ValueError(f'load only npy or pkl but got {fp}')
        
        if data.get('property', None) is not None:
            data[self.meta_key] = data.pop('property')
            # data['instance_mapping'] = data['property'].pop('instances')    
            data[self.meta_key]['original_affine'] = property2affine(data[self.meta_key]) # 4x4
            data[self.meta_key]['filename_or_obj'] = data[self.meta_key]['seg_file']
        return data

get_valid_mask = lambda data, max_label : np.where((data > max_label) | (data < 0), 0, data)
array_zyx2xyz = lambda arr: arr.transpose(0, 3, 2, 1)
def convert_coord_nn(bbox_c6, is_nn_order = True, zyx2xyz = True):
    assert len(bbox_c6) == 6
    # bboxes: [x1, y1, x2, y2, z1, z2]
    if is_nn_order:
        x1, y1, x2, y2, z1, z2 = bbox_c6
    else:
        x1, y1, z1, x2, y2, z2 = bbox_c6

    if zyx2xyz: 
        x1, y1, z1, x2, y2, z2 = z1, y1, x1, z2, y2, x2
    return (x1, y1, z1, x2, y2, z2)

def property2affine(property_dict, show_result = False):
    # keys='spacing_after_resampling' , 'itk_direction', 'itk_origin'
    spacing = np.array(property_dict['spacing_after_resampling'])[::-1]
    spacing[:2] *= -1
    direction = np.array(property_dict['itk_direction'])
    origin = np.array(property_dict['itk_origin'])
    affine_3x3 = direction.reshape(3, 3) * spacing
    # image_3d = image_3d.transpose(2, 1, 0)
    affine_mat_raw = np.eye(4)
    affine_mat_raw[:3,:3] = affine_3x3
    # for i in range(3): affine_matrix[i, i]  = spacing[i]
    for i in range(3): affine_mat_raw[i, -1] = origin[i]
    if show_result: print('[affine]\n', affine_mat_raw)
    return affine_mat_raw

@PIPELINES.register_module()   
class InstanceBasedCrop:
    """
    # boxes :	 [[ 23 314  33 341  76  91]
        #         [166 164 185 211 377 414]
        #         [141 159 162 225 388 442]
        #         [117 175 138 234 427 462]
        #         [101 201 121 266 451 478]]
    # instances :	 [1, 2, 3, 4, 5]
    # labels :	 [0, 0, 0, 0, 0]
    """
    def __init__(self, 
                keys, 
                patch_size, 
                instance_key = 'instance_ix', 
                img_key = None,
                pyramid_scale = None,
                verbose = False
                ) -> None:
        self.keys = keys
        self.instance_ix = instance_key
        self.img_key = keys[0] if img_key is None else img_key
        self.patch_size = patch_size
        self.pyramid_scale = pyramid_scale
        self.verbose = verbose

    def __call__(self, data) -> Any:
        data = dict(data)
        image_shape = data[self.img_key].shape[-3:]
        self.patch_size = fall_back_tuple(self.patch_size, default=image_shape)
        instance_ix = data.get(self.instance_ix, -1)
        # get instance bbox
        bbox = None if instance_ix < 0 else data['bboxes']['boxes'][instance_ix - 1]
        if self.verbose: print(f'\n[InsCrop] {self.instance_ix} bbox {bbox} imgshape {image_shape}')
        # bbox to center
        if bbox is None:
            center_c3 = self.random_background_center(image_shape, self.patch_size)
        else:
            center_c3 = self.random_foreground_center(bbox)
        
        # check if box match image coordinates
        # 
        if self.verbose: print(f'[FindCenter] instance {instance_ix} bbox{bbox} center {center_c3}')
        # create cropper    
        cropper =  SpatialCrop_(roi_center=tuple(center_c3), roi_size=self.patch_size)
        # crop key by key
        for k in self.keys:
            prior_shape = data[k].shape
            data[k] = cropper(data[k])
            data['img_meta_dict'][f'cropshape'] = tuple(data[k].shape[1:])
            if self.verbose: print_tensor(f'[Crop] {k} pre {prior_shape} after', data[k])

        # validate mask
        inst2cls = data['img_meta_dict']['instances']
        if len(inst2cls) > 0:
            max_label = max([int(a) for a in  data['img_meta_dict']['instances']])
        else: max_label = 0
        data['seg'] = get_valid_mask(data['seg'], max_label)
        return data

    def random_background_center(self, image_shape, patch_shape, verbose = False):
        assert len(image_shape) == len(patch_shape)
        center_bg = []
        for dim, (img_size, p_size) in enumerate(zip(image_shape, patch_shape)):
            cen_range = (p_size//2, img_size - p_size//2)
            if verbose: print(f'\t[BGCenter] {dim} img {img_size} patch {p_size} cen {cen_range}')
            if len(set(cen_range)) == 1: 
                rand_center = cen_range[0]
            else:
                rand_center = random.randrange(min(cen_range), max(cen_range))
            center_bg.append(rand_center)
        return center_bg

    def random_foreground_center(self, bbox_c6, is_nn_order = True, 
                                zyx2xyz = True,  verbose = False): # 
        """
        nnDet: dim order zyx, image permute to xyz, here bbox also should be permuted 
               z reordered

        Args:
            bbox_c6 ([type]): [description]
            image_shape ([type]): [description]
            is_nn_order (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        assert len(bbox_c6) == 6
        # bboxes: [x1, y1, x2, y2, z1, z2]

        x1, y1, z1, x2, y2, z2 = convert_coord_nn(bbox_c6, is_nn_order,  zyx2xyz)

        # if isinstance(z_size, int): 
        #     z1, z2 = max(z_size - z2, 0), max(z_size - z1, 0)
        if verbose: print(f'\t[FGCenter] xyz1,xyz2 ', x1, y1, z1, x2, y2, z2)
        center_fg = []
        for center_min, center_max in zip((x1, y1, z1), (x2, y2, z2)):
            rand_center = random.randrange(center_min, center_max)
            center_fg.append(rand_center)
        return center_fg