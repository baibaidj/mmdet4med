import os
import os.path as osp
import numpy as np
import nibabel as nb
from PIL import Image
from pathlib import Path
import time, json, random
import SimpleITK as sitk
from .png_rw_ import read_png2array, IO4Png
import pdb, codecs, re

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def save2json(obj, data_rt, filename, indent=4, sort_keys=True):
    assert 'json' in filename
    fp = osp.join(data_rt, filename)
    with open(fp, 'w', encoding='utf8') as f:
        json.dump(obj, f, sort_keys=sort_keys, ensure_ascii=False, indent=indent)
    return fp

def load2json(json_fp):
    assert osp.exists(json_fp)
    data = dict()
    if os.path.exists(json_fp):
        with open(json_fp, 'r') as f:
            data = json.load(f)
    return data


def save_string_list(file_path, l, is_utf8=False):
    """
    Save string list as mitok file
    - file_path: file path
    - l: list store strings
    """
    file_path = str(file_path)
    l = [l] if type(l) is not list else l
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    nb_l = len(l)
    for i, item in enumerate(l):
        item = str(item)
        line = item if i == nb_l- 1 else (item + '\n')
        f.write(line)
    f.close()


def remove_white_space(strings):
    try:
        result = re.sub(r'[\(\)\t\n\s]', '', strings)
    except:
        result = strings
    return result


def load_string_list(file_path, is_utf8=False):
    """
    Load string list from mitok file
    """
    try:
        if is_utf8:
            f = codecs.open(file_path, 'r', 'utf-8')
        else:
            f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print('open error %s' % file_path)
        return None
    else:
        return l

def convert_label(label, label_mapping = None, inverse=False, value4outlier = 0):
    if label_mapping is None:
        return label
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    
    if not inverse:
        max_value = max([v for _, v in label_mapping.items()])
        label[label > max_value] = value4outlier
    return label

def open_mask_random(fp):
    """
    more than one mask may exist for a image, as 160_{0,1,9}.png
    during training, randomly load one 
    """
    fp = str(fp)
    pil_load = lambda fn: np.array(Image.open(fn))
    # fp = Path(fp)
    fn = fp.split('/')[-1].split('_')[0]
    fn_dir = '/'.join(fp.split('/')[:-1])
    fp_by_doc = lambda i : '/'.join([fn_dir, str('%s_%d.png'%(fn, i))])

    if osp.exists(fp_by_doc(9)):
        return pil_load(fp)
    else:
        fp_candidates = [fp_by_doc(i) for i in (0,1)]
        if np.random.random() > 0.5: fp_candidates = fp_candidates[::-1]
        for cur_fp in fp_candidates:
            if osp.exists(cur_fp):
                return pil_load(cur_fp)
    return pil_load(fp)


pil_load = lambda fn: np.array(Image.open(str(fn)))
get_s_ix = lambda fn: int(fn.stem.split('/')[0])

class LoadMultipleSlices(object):
    """
    given the file path of a slice and number of slices,
    load a volume centered at the given slice
    The slices will be stacked in the last dimension to accommondate later transformations
    
    tg=target
    """
    def __init__(self, fp, nb_slices = 3, max_interval = 1, 
                tg_position = 'mid', verbose = False, pid_info = None,
                ) -> None:
        assert tg_position in ('mid', 'head', 'tail')
        self.fp = Path(fp)
        self.file_dir = self.fp.parent
        self.fn = self.fp.name
        self.nb_slices = nb_slices
        self.max_interval = max_interval
        self.tg_position = tg_position
        self.verbose = verbose
        # print(self.file_dir)
        self.all_fns_dict = None if pid_info is None else pid_info[str(self.file_dir)]

        self.is_mask = '_' in self.fp.stem
        self.image_class = [int(a) for a in self.fp.stem.split('_')[1].split('-')] if self.is_mask else None

        self.Loader = IO4Png(verbose= verbose)
        # if self.verbose: print('fn %s, is_mask %s, class %s' %(self.fn, self.is_mask, self.image_class))


    @property
    def tg_ix(self):
        return int(self.fp.stem[:3])

    def get_pre_post_nb(self):
        if self.tg_position == 'mid':
            pre_nb = int((self.nb_slices - 0.001) // 2)
            post_nb = self.nb_slices - pre_nb - 1
            return pre_nb, post_nb
        elif self.tg_position == 'head':
            return 0, self.nb_slices -1
        elif self.tg_position == 'tail':
            return self.nb_slices -1, 0
        else:
            raise RuntimeError('target_position can only assume "mid", "head" and "tail" but inputed %s' %self.tg_position)

    @property
    def skip_step(self):
        if self.max_interval > 1:
            return random.randint(1, self.max_interval)
        else:
            return 1

    def get_fps2load(self): 

        if self.nb_slices == 1:
            return [self.fp], 0
        tg_ix = self.tg_ix
        pre_nb, post_nb = self.get_pre_post_nb()

        if self.all_fns_dict is None:
            all_fns = os.listdir(self.file_dir)
            all_fns_dict = {int(str(f)[:3]) : f for f in all_fns}
        else:
            all_fns_dict = self.all_fns_dict
            assert len(all_fns_dict) > 0
        nb_slices = max(list(all_fns_dict)) + 1 # slice index start from 0
        # tg_ix = 398

        pre_ixs = [max(tg_ix - (pre_nb - i) * self.skip_step , 0) for i in range(pre_nb)]
        post_ixs = [min(tg_ix + (i + 1) * self.skip_step, nb_slices - 1) for i in range(post_nb)]
        ixs = pre_ixs + [tg_ix] + post_ixs

        if self.verbose: print(self.fn, ixs)

        fps = [self.file_dir/all_fns_dict[i] if i in all_fns_dict else self.fp for i in ixs ]
        
        ocurrance = lambda x: sum([1 for a in ixs if a == x]) - 1
        z_bound_count = -1* max(0, ocurrance(0)) + 1 * max(0, ocurrance(nb_slices - 1))
        if self.verbose: print('\ttg_ix:%d\ttotal%d\tipch:%d\tzbound:%d' %(
                        tg_ix, nb_slices, self.nb_slices, z_bound_count))
        return fps, z_bound_count
    
    def _load_images(self, fps, load_func):
        center_fp = fps[len(fps)//2]
        x_list = []
        for fp in fps:
            fn_i = center_fp if not osp.exists(fp) else fp
            # if self.input_duplicates:
            x = load_func(fn_i)
            x_list.append(x)
            # print('\n')
            # print('tg %s actual %s' %(tg_ix, actual_ixs))
        x = np.stack(x_list, axis=-1)
        return x
    
    def load(self, is_z_first = False, use_med_view = False):
        fps, z_bound_count = self.get_fps2load() 
        load_func = pil_load #open_mask_random if self.is_mask else
        image = self._load_images(fps, load_func)
        
        if use_med_view:
            fps_med = [str(fp).replace('image_links', 'image_links_med') for fp in fps]
            assert all([osp.exists(f) for f in fps_med]), \
                'Medview file not exists %s' %str(fps_med[self.nb_slices//2])
            image_med = self._load_images(fps_med, load_func)
            image = np.concatenate([image, image_med], axis = -1)
        if is_z_first:
            image = np.moveaxis(image, -1, 0)
        return image, z_bound_count

    
    def load_spng(self,  is_z_first = False, dtype = np.int16):
        fps, z_bound_count = self.get_fps2load()
        center_fp = fps[len(fps)//2]
        x_list = []
        y_list = []
        spacing_slice = None
        for i, fp in enumerate(fps):
            fn_i = center_fp if not osp.exists(fp) else fp
            # if self.input_duplicates:
            x, y, spacing= self.Loader.read(str(fn_i), dtype) #read_png2array(fn_i)
            x_list.append(x)
            y_list.append(y)
            if i == len(fps)//2:
                spacing_slice = spacing
            # print('\n')
            # print('tg %s actual %s' %(tg_ix, actual_ixs))
        image = np.stack(x_list, axis=-1)
        mask = np.stack(y_list, axis = -1)

        if is_z_first:
            image = np.moveaxis(image, -1, 0)
            mask = np.moveaxis(mask, -1, 0)
        return image, mask, z_bound_count, spacing_slice

def mkdir(path):
    # credit goes to YY
    # 判别路径不为空
    path = str(path)
    if path != '':
        # 去除首空格
        path = path.rstrip(' \t\r\n\0')
        if '~' in path:
            path = os.path.expanduser(path)
        # 判别是否存在路径,如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)



view2permute = {'saggital': (0, 1, 2),
                'coronal': (2, 0, 1),
                'axial': (1, 2, 0)
                }

# normally z axis should be in first dimension
view2axis = {'saggital': 'xzy',
            'coronal': 'yzx',
            'axial': 'zyx',
            None: None,
            }
# nii dataset always assume xyz dimension order
axis_order_map = {'xzy' : (0, 2, 1), 
                  'zyx': (2, 1, 0), 
                  'yzx': (1, 2, 0),
                  None: None
}

axis_reorder_map = {'xzy' : (0, 2, 1),
                    'zyx' : (2, 1, 0),
                    'yzx' : (2, 0, 1),
                    None: None}

# xyz-> xcyz -> x=0,c=1,y=2,z=3
axis_reorder_map4d = {'xzy' : (0, 1, 3, 2), #xczy xcyz
                      'zyx' : (3, 1, 2, 0), #zcyx xcyz
                      'yzx' : (3, 1, 0, 2)} #yczx xcyz
# image and mask are already saved as nii

def affine_in_sitk_obj(sitk_obj):
    spacing = np.array(sitk_obj.GetSpacing(), dtype = np.float32).reshape(1, 3)
    for i in range(2): spacing[0, i] *= -1 # LPS+ > RAS+, only xy are reversed
    origin = np.array(sitk_obj.GetOrigin(), dtype = np.float32)
    direction = np.array(sitk_obj.GetDirection())
    affine_3x3 = direction.reshape(3, 3) * spacing
    # image_3d = image_3d.transpose(2, 1, 0)
    affine_mat_raw = np.eye(4, dtype = float )
    affine_mat_raw[:3,:3] = affine_3x3
    # for i in range(3): affine_matrix[i, i]  = spacing[i]
    for i in range(3): affine_mat_raw[i, -1] = origin[i]
    return affine_mat_raw

class IO4Nii(object):

    """
    
    Nibabel images always use RAS+ output coordinates, where 
    Right, anterior and superior direction in the patient's perspective is regarded as positive.
    reference: https://nipy.org/nibabel/coordinate_systems.html
    

    If Anatomical Orientation Type (0010,2210) is absent or has a value of BIPED, 
    the x-axis is increasing to the left hand side of the patient. 
    The y-axis is increasing to the posterior side of the patient. 
    The z-axis is increasing toward the head of the patient.
    This is LPS+ coordinates system. 
    reference: https://dicom.innolitics.com/ciods/ct-image/general-series/00102210

    Normally, 0010,2210 tag is absent in the dicom of human subjects, 
    which means that most dicom uses LPS+ coordinates system. 
    When conversion to nifti, we need to transform LPS+ to RAS+ 
    by setting the pixel spacing on axial/xy plane to negative. 

    """
    @staticmethod
    def read(img_nii_fp, verbose = True, axis_order = None, dtype = None, 
             row_first = True):
 
        assert axis_order in list(axis_order_map)

        try:
            if verbose: print('NIBABEL')
            nii_obj = nb.load(img_nii_fp)
            affine_mat_raw = nii_obj.affine.astype(np.float32)
            image_3d_raw = nii_obj.get_fdata()
            # permute_order = axis_order_map[axis_order]
            # image_3d_raw = image_3d_raw.transpose(permute_order)
            # axis_order = None

        except EOFError or OSError:
            # print('[Error] IO4Nii.read ', img_nii_fp)
            print('[IO4Nii] sitk read', img_nii_fp)
            sitk_obj = sitk.ReadImage(img_nii_fp)
            image_3d_raw = sitk.GetArrayFromImage(sitk_obj)
            image_3d_raw = np.transpose(image_3d_raw, (2,1,0))
            affine_mat_raw = affine_in_sitk_obj(sitk_obj)

        # In the patient-centered (world) coordinate system, where x pointing from right to left, y from anterior to posterior, z from feet to head
        # the first 3 rows of the affine matrix are base vectors/directions for xyz axis of the tensor.  
        # cosine 0 degree = 1, cosine 90 degree = 0, cosine 180 degree = -1
        # for instance, the 1st row of (1, 0, 0) mean the x-axis of the tensor is the same as the world coord system. 
        # while the 3st row of (0, 0, -1) mean the z-axis of the tensor is the x-axis reversed of the world coord system.

        # Transform all 
        # [[ 0.          0.         -0.93945312  0.        ]
        #  [ 0.         -0.93945312  0.          0.        ]
        #  [-1.          0.          0.          0.        ]
        #  [ 0.          0.          0.          1.        ]]

        # To, both affine_matrix and image_tensor
        # [[-0.93945312  0.          0.          0.        ]
        #  [ 0.         -0.93945312  0.          0.        ]
        #  [ 0.          0.         -1.0         0.        ]
        #  [ 0.          0.          0.          1.        ]]

        # between axis order
        xyz_dim_index = list(np.argmax(np.abs(affine_mat_raw[:3, :3]), axis= 0 if row_first else 1))
        image_3d_xyz = np.transpose(image_3d_raw, axes = xyz_dim_index)
        affine_matrix = np.copy(affine_mat_raw)
        for i, d in enumerate(xyz_dim_index): 
            if row_first:
                affine_matrix[i, i], affine_matrix[d, i] = affine_mat_raw[d, i], affine_mat_raw[i, i]
            else:
                affine_matrix[i, i], affine_matrix[i, d] = affine_mat_raw[i, d], affine_mat_raw[i, i]
        affine_matrix = np.round(affine_matrix, decimals = 6)

        # within axis direction
        xyz_sign = [int(np.sign(affine_mat_raw[i,i] * (-1 if i <2 else 1))) for i in range(3)]
        xyz_sign = [a if a!= 0 else 1 for a in xyz_sign]
        image_3d_xyz = image_3d_xyz[::xyz_sign[0], ::xyz_sign[1], ::xyz_sign[2]]
        if verbose: 
            print('\n\n[IO4Nii] affine raw\n', affine_mat_raw)
            print('[IO4Nii] affine xyz\n', affine_matrix)
            print('[IO4Nii] xyz dim index', xyz_dim_index)
            print('[IO4Nii] xyz sign', xyz_sign)
            print_tensor('[IO4Nii] image raw', image_3d_raw)
            print_tensor('[IO4Nii] image xyz', image_3d_xyz)

        permute_order = axis_order_map[axis_order]
        if permute_order is None:image_3d = image_3d_xyz
        else:  image_3d = image_3d_xyz.transpose(permute_order)
        if dtype is not None:
            image_3d = image_3d.astype(dtype)
        return image_3d, affine_matrix

    @staticmethod
    def read_ww(img_nii_fp, axis_order = 'zyx',  
                ww = 400, wc = 50, is_uint8 = True, verbose = True):
        
        image_new, affine_matrix = IO4Nii.read(img_nii_fp, 
                                                axis_order = axis_order ,
                                                verbose = verbose)
        if isinstance(ww, int) and isinstance(wc, int):
            image_new = adjust_ww_wl(image_new, ww, wc, is_uint8)
        return image_new, affine_matrix

    @staticmethod
    def read_shape_xyz(img_nii_fp, verbose = False):
        nii_obj = nb.load(img_nii_fp)
        image_shape_xyz = nii_obj.header.get_data_shape()
        # image_shape_zyx = image_shape_xyz[::-1]
        if verbose: print(image_shape_xyz)
        return image_shape_xyz

    @staticmethod
    def write(mask_3d, store_root, file_name, affine_matrix = None, nii_obj= None,
                is_compress = True, axis_order = None):
        """
        must transform the input tensor to xyz, which then can be properly saved.
        and affine matrix are diagonal 
        将输入图像存储为nii
        输入维度是z, r=y, c=x
        输出维度是x=c, y=r, z
        :param mask_3d:
        :param store_root:
        :param file_name:
        :param nii_obj:
        :return:
        """
        extension = 'nii.gz' if is_compress else 'nii'
        permute_order = axis_reorder_map[axis_order]
        if permute_order is not None:
            mask_3d = mask_3d.transpose(permute_order)
        # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
        store_path = osp.join(store_root, '.'.join([file_name, extension]))
        if nii_obj is None:
            if affine_matrix is None: 
                affine_matrix = np.eye(4)
                for i in range(2): affine_matrix[i, i] = -1
            xyz_dim_index = list(np.argmax(np.abs(affine_matrix[:3, :3]), axis= 1))
            xyz_sign = [int(np.sign(affine_matrix[d,i] * (-1 if i <2 else 1))) for i, d in enumerate(xyz_dim_index)]
            # print(affine_matrix, x_col_sign, y_row_sign)
            nb_ojb = nb.Nifti1Image(mask_3d[::xyz_sign[0], ::xyz_sign[1], ::xyz_sign[2]], affine_matrix)
        else:
            nb_ojb = nb.Nifti1Image(mask_3d, nii_obj.affine, nii_obj.header)

        nb.save(nb_ojb, store_path)
        return store_path

    @staticmethod
    def write4d(mask_4d, store_root, file_name, affine_matrix = None, nii_obj= None,
                is_compress = True, axis_order = 'zyx'):
        """
        将输入图像存储为nii 
        输入维度是z, r=y, c=x
        输出维度是x=c, y=r, z
        :param mask_4d: assume the channel dim is the last
        :param store_root:
        :param file_name:
        :param nii_obj:
        :return:
        """
        extension = 'nii.gz' if is_compress else 'nii'
        permute_order = axis_reorder_map[axis_order]
        if permute_order is not None:
            mask_4d = mask_4d.transpose(permute_order + (3, ))
        # mask_3d = mask_3d[::-1,::-1,:] # be cautious to uncomment this line
        store_path = osp.join(store_root, '.'.join([file_name, extension]))
        if nii_obj is None:
            if affine_matrix is None: affine_matrix = np.eye(4) 
            x_col_sign = int(np.sign(-1 * affine_matrix[0,0]))
            y_row_sign = int(np.sign(-1 * affine_matrix[1,1]))
            # print(affine_matrix)
            nb_ojb = nb.Nifti1Image(mask_4d[::x_col_sign, ::y_row_sign, ...], affine_matrix)
        else:
            nb_ojb = nb.Nifti1Image(mask_4d, nii_obj.affine, nii_obj.header)

        nb.save(nb_ojb, store_path)
        return store_path
        
# @staticmethod
def adjust_ww_wl(image, ww = -600, wc = 1600, is_uint8 = True, force2postive = True):
    """
    image is forced to be positive to stay compatible with the following 
    image processing operations using cv2, e.g. padding, elastic transformation and rotation. 
    调整图像得窗宽窗位
    :param image: 3D图像
    :param ww: 窗宽
    :param wc: 窗位
    :return: 调整窗宽窗位后的图像
    """
    min_hu = wc - (ww/2)
    max_hu = wc + (ww/2)
    new_image = np.clip(image, min_hu, max_hu)#np.copy(image)
    if is_uint8:
        new_image -= min_hu
        new_image = np.array(new_image / ww * 255., dtype = np.uint8)
    
    if force2postive and min_hu < 0:
        new_image -= min_hu
    
    return new_image


class ImageDrawerDHW(object):
    """
    start with a image tensor, such as CT volume in D, H, W
    give an index of slice and nb_channels, return a subset of slices centered on that index

    index from the first dimension
    put the indexing dim to the last dim as channels 
    """
    def __init__(self, image_tensor, nb_channels = 3, dim0_to_last = True, skip_step = 1,
                fp = '', verbose = False) -> None:

        self.image = image_tensor
        self.nb_channels = nb_channels
        self.dim0_to_last = dim0_to_last
        self.skip_step = skip_step
        self.fp = fp
        self.verbose = verbose
    
    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        assert isinstance(index, int)
        assert (index >= 0) and (index < len(self)), print_tensor(f'fp"{self.fp} ix: {index}', self.image )
        pre_nb = int((self.nb_channels - 0.001) // 2)
        post_nb = self.nb_channels - pre_nb - 1
        tg_ix = index
        nb_slices = len(self)
        # tg_ix = 398
        pre_ixs = [max(tg_ix - (pre_nb - i) * self.skip_step , 0) for i in range(pre_nb)]
        post_ixs = [min(tg_ix + (i + 1) * self.skip_step, nb_slices - 1) for i in range(post_nb)]
        ixs = pre_ixs + [tg_ix] + post_ixs
        ip_slices = self.image[ixs, ...]
        if self.dim0_to_last: ip_slices = np.moveaxis(ip_slices, 0, -1)
        return ip_slices, ixs



# # pair image and mask paths, subdir
class Path4TCIA(object):
    """
    given the subdir file of a dataset
    load the absolute path of each data file
    do data split
    
    """
    # data_rt = Path('/lung_general_data/shidj/public/TCIA-pancreas')
    # img_dir = data_rt/'Pancreas_nii'
    # mask_dir = data_rt/'TCIA_pancreas_labels-02-05-2017'
    # store_dir = 'pancreas2train'
    # train_fn = 'train_png_pairs.txt'
    # subdir_fn = 'entire_sub_dir_list.txt'
    # ww, wc, is_uint8 = 400, 40, True
    # data_split = (8, 2)

    def __init__(self, data_rt, img_dir = 'Pancreas_nii', 
                 mask_dir = 'TCIA_pancreas_labels-02-05-2017', 
                 store_dir = 'pancreas2train', 
                 subdir_fn = 'entire_sub_dir_list.txt', 
                #  train_fn = 'GT_train_pairs.txt',
                 ww = 400, wc = 40, is_uint8 = True,
                 data_split = (8, 2),
                 new_spacing = None,
                 split_names = ('train', 'val', 'test'),
                 seed = 42

                 ) -> None:

        self.data_rt = data_rt
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.store_dir = store_dir
        self.subdir_fn = subdir_fn
        # self.train_fn = train_fn
        self.ww, self.wc, self.is_uint8 = ww, wc, is_uint8
        self.data_split = data_split
        self.new_spacing = new_spacing
        self.split_names = split_names
        self.seed = seed
        assert len(self.data_split) <= len(self.split_names)

        self.subdirs = load_string_list(self.data_rt/self.img_dir/self.subdir_fn)
    

    def split_train_val(self, data_split, seed = 42, verbose = False):
        # split to train and val
        nb_pids = len(self.subdirs)
        new_indexs = list(range(nb_pids))
        nb_split = len(data_split)
        print('\t@@@SEED: %d@@@'%seed)
        if seed:
            random.seed(seed)
            random.shuffle(new_indexs)
        pid_shuffle_list = [self.subdirs[i] for i in new_indexs]

        breakpoints = [int(data_split[n]/sum(data_split) * nb_pids) for n in range(nb_split - 1)]
        bp_cumulate = [0]
        for bp in breakpoints:
            bp_cumulate.append(bp_cumulate[-1] + bp)
        bp_cumulate.append(nb_pids)
        print('Num pids %d, split %s, ix points %s' %(nb_pids, data_split, bp_cumulate))
        # split_slices = [  for i in range(nb_split)]
        split_ix_list = [pid_shuffle_list[bp_cumulate[p]:bp_cumulate[p+1]] for p in range(nb_split)]

        # pid_shuffle_list[:train_end], pid_shuffle_list[train_end:] 
        return split_ix_list

    def get_nii_path_pairs(self):
        pid_data_dict = {}
        for sd in self.subdirs:
            idx = sd.split(os.sep)[0].split('_')[-1]  
            img_path = self.data_rt/self.img_dir/sd/'image.nii.gz'
            mask_path = self.data_rt/self.mask_dir/str('label%s.nii.gz'%idx)
            pid_data_dict[sd] = {'img_nii_fp': img_path, 'gt_nii_fp': mask_path}

        return pid_data_dict


    def main(self, is_test = True):

        pid_data_dict = self.get_nii_path_pairs()

        split_ix_list = self.split_train_val(self.data_split, seed = self.seed)

        split_data = {self.split_names[i]: split_ix_list[i] for i in range(len(self.data_split))} 
        # split_data = {'train': train_pids, 'val': val_pids}
        data_ext = 'png' if 'png' in self.store_dir else 'nii'
        
        for split, pids in split_data.items():
            if data_ext == 'png':
                samples = self.save2png_by_group(pid_data_dict, pids, is_write= not is_test)
            else:
                if self.new_spacing is None:
                    samples = self.soft_link_nii_by_group(pid_data_dict, pids, 
                                                split_name = split, 
                                                is_test = is_test)
                else:
                    samples = self.save2nii_norm_by_group(pid_data_dict, pids,
                                                split_name= split, 
                                                is_test = is_test )
            
            split_fn = str('%s_%s.txt' %(split, data_ext))
            save_string_list(self.data_rt/self.store_dir/split_fn, samples)

    

# class SpatialNormer():
#     """
#     given old-spacing and new-spacing, 
#     transform old shape to new shape
#     the dimensions for spacing and shape should be the same
#     default is z, y, x; or axial, saggital, coronal 
    
#     """
#     def __init__(self, affine_matrix, new_spacing, verbose = False):
#         assert affine_matrix.shape == (4,4)
#         assert len(new_spacing) ==3

#         self.affine_matrix = affine_matrix
#         self.old_spacing = [affine_matrix[2-i, 2-i] for i in range(3)] # z, y, x
#         self.new_spacing = [new_spacing[a] * np.sign(self.old_spacing[a]) for a in range(3)]

#         self.verbose = verbose
#         if self.verbose: print('\n\nold affine\n', affine_matrix)

#     @property
#     def x_col_sign(self): return int(-1 * np.sign(self.affine_matrix[0,0]))

#     @property
#     def y_row_sign(self): return int(-1 * np.sign(self.affine_matrix[1,1]))

#     @property
#     def z_slice_sign(self): return int(np.sign(self.affine_matrix[2,2]))

#     @property
#     def new_affine(self):
#         new_affine_matrix = np.copy(self.affine_matrix)
#         for i in range(3):
#             new_affine_matrix[2-i, 2-i] = self.new_spacing[i]
        
#         if self.verbose: print('new affine\n', new_affine_matrix)
#         return new_affine_matrix

#     @property
#     def trans_ratio(self):
#         ratio = [ abs(self.old_spacing[a]/self.new_spacing[a]) for a 
#                  in range(3)]
#         if self.verbose: print('trans ratio\n', ratio)
#         return ratio
    
#     def new_shape(self, old_shape):
#         new_shape = [int(old_shape[a] * self.trans_ratio[a]) for a in range(3)]
#         if self.verbose: print('new shape\n', new_shape)
#         return new_shape

#     def transform(self, input_image, mode = 'nearest'):
#         if self.verbose: print_tensor('input\n', input_image)
#         try:
#             input_tensor = torch.from_numpy(input_image.copy())
#         except TypeError:
#             input_image = np.array(input_image, dtype = np.uint8)
#             input_tensor = torch.from_numpy(input_image.copy())
            
#         np_dtype = input_image.dtype 
#         img_d, img_h, img_w = input_image.shape
#         input_tensor = input_tensor.view(1, 1, img_d, img_h, img_w).float()
#         resize_tensor = F.interpolate(input_tensor, self.new_shape(input_image.shape), mode=mode).data[0, 0]
        
#         if self.verbose: print_tensor('post trans\n', resize_tensor)
#         resize_tensor = np.array(resize_tensor, dtype = np_dtype)
#         return resize_tensor

