import numpy as np
from numpy.lib.type_check import imag
from .load_nn import load_pickle, Any
from ..builder import PIPELINES
from .io4med import print_tensor, random
import sys, pdb

from monai.transforms.croppad.array_ import SpatialCrop_, SpatialMultiPyramidCrop_
from monai.utils import fall_back_tuple, ensure_tuple_rep

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
