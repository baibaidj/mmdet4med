"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

import os
import SimpleITK as sitk
import numpy as np
import torch
import pydicom
import time
import torch.nn.functional as F
import zipfile
import subprocess
import multiprocessing


print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())

def get_git_revision_hash():
    try:
        info = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        info = 'no git revision'
    return info


class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        self.val = val
        self.sum += val * ncount
        self.count += ncount
        self.avg = self.sum / self.count


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def list_subfiles(path):
    _files = []
    list = os.listdir(path)
    tmpfiles = []
    for i in range(0, len(list)):
        path_ = os.path.join(path, list[i]).replace('\\', '/')
        if os.path.isfile(path_):
            tmpfiles.append(path_)
    if len(tmpfiles) > 0:
        _files.extend(tmpfiles)
    return _files


def list_all_files(path):
    _files = []
    list = os.listdir(path)
    tmpfiles = []
    for i in range(0, len(list)):
        path_ = os.path.join(path, list[i]).replace('\\', '/')
        if os.path.isdir(path_):
            _files.extend(list_all_files(path_))
        if os.path.isfile(path_):
            tmpfiles.append(path_)
    if len(tmpfiles) > 0:
        _files.extend(tmpfiles)
    return _files


def list_subfolders(path):
    folders = os.listdir(path)
    saved_folders = []
    for i in range(len(folders)):
        cfolders = path + folders[i] + '/'

        if cfolders.endswith('.DS_Store/'):
            continue
        if os.path.isdir(cfolders):
            saved_folders.append(cfolders)
    return saved_folders


def list_all_folders(path):
    sub_folders = []
    for root, dirs, files in os.walk(path, topdown=False):
        if len(files) > 0 and len(dirs) == 0:  # None sub folder include
            root = root.replace('\\', '/')
            s = root if root.endswith('/') else root + '/'
            sub_folders.append(s)
    return sub_folders


def get_folder_names(path):
    folders = list_all_folders(path)
    folders = [f.split('/')[-2] for f in folders]
    return folders


def zip_file(src_dir):
    zip_name = src_dir + '.zip'
    z = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        fpath = dirpath.replace(src_dir, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
            print('==压缩成功==')
    z.close()


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip, zip_src:', zip_src, 'dst_dir:', dst_dir)


def unzip_file2(srcDir, destDir):
    if not os.path.isdir(destDir) and os.path.isdir(srcDir):
        destDir = srcDir

    if os.path.isdir(srcDir) and os.path.isdir(destDir):
        'unzip file'
        for file_name in os.listdir(srcDir):
            path = os.path.join(srcDir, file_name)
            if zipfile.is_zipfile(path):
                z = zipfile.ZipFile(path, 'r')
                z.extractall(destDir)
                z.close()
                # os.remove(path)
        else:
            'change charset'
            for root_path, dir_names, file_names in os.walk(destDir):
                # print("xx", file_names)
                for fn in file_names:
                    path = os.path.join(root_path, fn)
                    if not zipfile.is_zipfile(path):
                        try:
                            fn = fn.encode('cp437').decode('gbk')
                            new_path = os.path.join(root_path, fn)
                            os.rename(path, new_path)
                        except Exception as e:
                            print('error:', e)
    else:
        raise Exception("src path is not a dir, ", srcDir, 'destDir: ', destDir)


def save_skimg(data, name, voxel_size=(1.0, 1.0, 1.0)):
    tmpImg = sitk.GetImageFromArray(data)
    tmpImg.SetSpacing(voxel_size)
    sitk.WriteImage(tmpImg, str(name) + '.nii.gz')


def load_skimg(path):
    data = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(data)
    return arr


def save_raw(data, name):
    str_name = data.tostring()
    str_path = open(str(name), 'wb')
    str_path.write(str_name)
    str_path.close()


def load_dicom(path, method=0):
    volumes = []
    if method == 0:
        reader = sitk.ImageSeriesReader()
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)
        for series_uid in series_IDs:
            dicom_names = reader.GetGDCMSeriesFileNames(path, series_uid)
            reader.SetFileNames(dicom_names)
            volume = reader.Execute()
            volumes.append(volume)
            return volumes
    else:
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness

        volume = np.stack([s.pixel_array for s in slices])

        pixel_presentation = slices[0][(0x0028, 0x0103)].value
        volume = volume.astype(
            np.uint16) if pixel_presentation == 0 else volume.astype(np.int16)

        volume = sitk.GetImageFromArray(volume)
        voxel_size = [float(slices[0][(0x0028, 0x0030)].value[0]), float(slices[0][(0x0028, 0x0030)].value[1]),
                      slice_thickness]
        volume.SetSpacing(voxel_size)
        volumes.append(volume)
        return volumes


def resize(array, size, mode, align_corners, GPU):
    ori_type = array.dtype
    shape_length = 5 - len(array.shape)
    assert shape_length >= 0
    for s in range(shape_length):
        array = array[np.newaxis, :]
    array = array.astype(np.float)
    tensor = torch.from_numpy(array)
    if GPU is True:
        tensor = tensor.cuda()
    tensor_new = F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
    array_new = tensor_new.cpu().numpy()
    for s in range(shape_length):
        array_new = array_new[0]
    array_new = array_new.astype(ori_type)
    return array_new


def z_score(array):
    m = np.mean(array)
    std = np.std(array)
    out = (array - m) / std
    return out


def write_dicom(ref_path, nii_arr):
    save_path = './dicom/'
    mkdir(save_path)

    # reference
    ref_series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(ref_path)
    if not ref_series_IDs:
        print("ERROR: given directory \"" + ref_path +
              "\" does not contain a DICOM series.")
        return 403
    ref_series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        ref_path, ref_series_IDs[0])
    ref_series_reader = sitk.ImageSeriesReader()
    ref_series_reader.SetFileNames(ref_series_file_names)

    ref_file = list_all_files(ref_path)[0]
    ref_image_reader = sitk.ReadImage(ref_file)
    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    ref_series_reader.MetaDataDictionaryArrayUpdateOn()
    ref_series_reader.LoadPrivateTagsOn()
    ref_image3D = ref_series_reader.Execute()

    # Modify the image (blurring)
    # filtered_image = sitk.DiscreteGaussian(ref_image3D)

    # read nii data
    # nii_data = sitk.ReadImage(nii_path)
    # nii_arr = sitk.GetArrayFromImage(nii_data)

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    # Copy relevant tags from the original meta-data dictionary (private tags are also
    # accessible).
    flt_tags_to_copy = ["0010|0010",  # Patient Name
                        "0010|0020",  # Patient ID
                        "0010|0030",  # Patient Birth Date
                        "0020|000D",  # Study Instance UID, for machine consumption
                        "0020|0010",  # Study ID, for human consumption
                        "0008|0020",  # Study Date
                        "0008|0030",  # Study Time
                        "0008|0050",  # Accession Number
                        "0008|0060"  # Modality
                        "0028|0030"  # Pixel Spacing
                        ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    spacing = ref_image3D.GetSpacing()
    direction = ref_image3D.GetDirection()
    BitsAllocated = int(ref_image_reader.GetMetaData("0028|0100"))
    PixelRepresentation = int(ref_image_reader.GetMetaData("0028|0103"))
    if BitsAllocated <= 16 and BitsAllocated > 8 and PixelRepresentation == 1:
        nii_arr = nii_arr.astype(np.int16)
    if BitsAllocated <= 16 and BitsAllocated > 8 and PixelRepresentation == 0:
        nii_arr = nii_arr.astype(np.uint16)
    if BitsAllocated > 16 and PixelRepresentation == 1:
        nii_arr = nii_arr.astype(np.int32)
    if BitsAllocated > 16 and PixelRepresentation == 0:
        nii_arr = nii_arr.astype(np.uint32)

    series_tag_values = [(k, ref_series_reader.GetMetaData(0, k)) for k in flt_tags_to_copy if
                         ref_series_reader.HasMetaDataKey(0, k)] + \
                        [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                         # Pixel Spacing
                         ("0028|0030", '\\'.join(
                             map(str, (spacing[0], spacing[1])))),
                         ("0020|000e", ref_series_reader.GetMetaData(0, "0020|000e") + \
                          modification_date + "." + modification_time),  # Series Instance UID
                         # Image Orientation (Patient)
                         ("0020|0037", '\\'.join(map(
                             str,
                             (direction[0], direction[3], direction[6], direction[1], direction[4], direction[7])))),
                         ("0008|103e",
                          ref_series_reader.GetMetaData(0, "0008|103e") + " Processed-SimpleITK")]  # Series Description

    nii_slice_nums = nii_arr.shape[0]
    ref_dcm_nums = ref_image3D.GetDepth()
    if nii_slice_nums != ref_dcm_nums:
        print("Error, the nums are not same!")
        return 405
    for i in range(ref_image3D.GetDepth()):
        image_slice_arr = nii_arr[i]  # filtered_image[:, :, i]
        image_slice = sitk.GetImageFromArray(image_slice_arr)
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime(
            "%Y%m%d"))  # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime(
            "%H%M%S"))  # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(
            str, ref_image3D.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
        image_slice.SetMetaData("0020,0013", str(i))  # Instance Number
        series_uid = image_slice.GetMetaData("0020|000e")
        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(save_path + series_uid + '_' + str(i) + '.dcm')
        writer.Execute(image_slice)
        return 0


def make_one_hot(input, num_classes):
    '''
    input: a tensor of shape [N, 1, *]
    num_classes: an int of number of class
    return: a tensor of shape [N, num_classes, *]
    '''
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(size=shape, dtype=torch.float, device=input.device)
    result = result.scatter_(1, input.long(), 1)
    return result


def dynamic_import(name: str, path: str):
    """dynamically import a lib.

    Args:
        name (str): The name of the lib
        path (str): The path of the lib

    Returns:
        [type]: The found module.
    """
    import importlib
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_parralel(func, iterables, *args, num_thread = 1, **kwargs):
    cpu_count = multiprocessing.cpu_count()
    if num_thread <= 0 or num_thread > cpu_count:
        num_thread = cpu_count

    is_dict = lambda x : isinstance(x, dict)

    def update_dict(holder, inputs):
        assert isinstance(holder, dict)
        assert isinstance(inputs, dict)
        
        for k, v in inputs.items():
            holder.setdefault(k, []).extend(v)
        # return holder

    def force2list(x):
        if isinstance(x, list):
            return x
        else:
            return list(x)
    sample_idx = sorted(list(iterables)) if is_dict(iterables) else list(range(len(iterables)))
    sample_num = len(sample_idx)
    kwargs['prcs_ix'] = 99
    print('[MultiProcess] Run parallel: num samples %d'  % sample_num)
    if num_thread > 1:  #
        per_part = sample_num // num_thread + (1 if sample_num % num_thread > 0  else 0)
        pool = multiprocessing.Pool(processes=num_thread)
        process_list = []
        for i in range(num_thread): #num_thread
            # if i in [18]:
            kwargs['prcs_ix'] = i
            start = int(i * per_part)
            stop = int((i + 1) * per_part)
            stop = min(stop, sample_num)
            print('[MultiProcess]thread=%d, start=%d, stop=%d' % (i, start, stop))
            if is_dict(iterables):
                inter_iterables = {k : iterables[k] for k in sample_idx[start:stop]}
            elif isinstance(iterables, pd.DataFrame):
                inter_iterables = iterables.iloc[start:stop]
            else:
                inter_iterables = [iterables[k] for k in sample_idx[start:stop]]
            this_proc = pool.apply_async(func, args=(inter_iterables, ) + args, kwds=kwargs)
            process_list.append(this_proc)
            # print('here')
        pool.close()
        pool.join()
        
        final_return = []

        print('[MultiProcess] Gather results')
        for pix, proc in enumerate(process_list):
            print(f'[MultiProcess] worker {pix}')
            return_objs = proc.get()
            if isinstance(return_objs, type(None)): return
            if not isinstance(return_objs, tuple):
                return_objs = (return_objs, )
            for rix, reo in enumerate(return_objs): 
                
                if pix == 0:
                    if is_dict(reo):
                        final_return.append(dict()) # assume that the keys are list
                    else:
                        final_return.append(list())
                if is_dict(reo):
                    update_dict(final_return[rix], reo)
                else:
                    final_return[rix].extend(force2list(reo))
            print('[MultiProcess] Done')
    else:
        final_return = func(iterables, *args, **kwargs)
        if not isinstance(final_return, tuple):
            final_return = (final_return, )

    return final_return
