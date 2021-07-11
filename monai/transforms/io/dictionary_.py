"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from typing import List
import h5py
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
import SimpleITK as sitk
import numpy as np


class LoadHDF5d(MapTransform):
    """
    Class for dictionary-based wrapper of IO loader of Hdf5.
    The cases are assumed storing in a single HDF5 file.
    """
    def __init__(
        self,
        keys: KeysCollection,
        h5_files: dict,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
    ) -> None:
        """hdf5 loader

        Args:
            keys (KeysCollection): keys to modify in input dict.
            h5_files (dict): h5 files list corresponding the keys.
        """
        self.keys = keys
        self.h5_files = h5_files
        self.meta_key_postfix = meta_key_postfix
        self.overwriting = overwriting
        self.h5_insts = None

    def get_meta_info(self, data, key):
        header = {}
        header["filename_or_obj"] = data[key]
        return header

    def __call__(self, data):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        # only open one time to speed up the reading.
        if self.h5_insts is None:
            self.h5_insts = {k: h5py.File(v, 'r') for k,v in self.h5_files.items()}
        d = dict(data)
        for k in self.keys:
            if k not in self.h5_files.keys():
                raise IOError(f'Can not find data, h5 file: {d[k]}, key: {k} ')

            h5 = self.h5_insts[k]
            arr = h5.get(d[k])
            if arr is None:
                raise IOError(f'Can not find data, h5 file: {d[k]}, key: {k} ')
            d[k] = arr[:]

            key_to_add = f"{k}_{self.meta_key_postfix}"
            if key_to_add in d and not self.overwriting:
                raise KeyError(f"Meta data with key {key_to_add} already exists and overwriting=False.")
            d[key_to_add] = self.get_meta_info(data, k)

        return d


class LoadItkd:
    """
    """

    def __init__(
        self,
        keys: KeysCollection
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        self.keys = keys
    def __call__(self, data):
        for key in self.keys:
            img = sitk.ReadImage(data[key])
            data[key] = sitk.GetArrayFromImage(img)
        return data


def dump_nifti_to_h5(h5_file: str,
                     nifti_files: List[str],
                     keys: List[str],
                     overwriting: bool = False,
                     dataset_opts: dict = {}):
    """dump nifti files list to a single h5 file.
    only the arr data is dumped, the spacing, origin, direction is dropped.

    Args:
        h5_file (str): h5 file to write to.
        nifti_files (List[str]): nifti file list.
        keys (List[str]): key list to access data.

    Returns:
        bool: True if success.
    """
    from os.path import exists
    import SimpleITK as sitk
    if not overwriting and exists(h5_file):
        raise IOError(f'The h5 file is exist: {h5_file}')
    assert len(nifti_files) == len(
        keys), 'The length of nifti_files and keys is not equal!'
    h5 = h5py.File(h5_file, 'w')
    for fname, key in zip(nifti_files, keys):
        img = sitk.ReadImage(fname)
        h5.create_dataset(key,
                          data=sitk.GetArrayFromImage(img),
                          **dataset_opts)
    h5.close()
    return True


class LoadDetInfo(MapTransform):
    """
    load detection information with LUNA format
    """

    def __init__(self, keys: KeysCollection, annotations, LUNA=False):
        super().__init__(keys)
        self.df = annotations
        self.LUNA = LUNA

    def __call__(self, data):
        d = dict(data)
        img_info = data['image_meta_dict']
        case_id = img_info['filename_or_obj'].split('/')[-1]
        if case_id.endswith('nii.gz'):
            case_id = case_id.split('_0000.nii.gz')[0]
        else:
            case_id = case_id.split('.')[0]

        # temp method, debug by tangzhe, to correct origin error from nib loader
        sitk_data = sitk.ReadImage(img_info['filename_or_obj'])

        if self.LUNA:
            info = self.df.loc[self.df['seriesuid'] == case_id]
        else:
            info = self.df.loc[self.df['case_id'] == case_id]

        nodule_nums = len(info)
        results = []
        for n in range(nodule_nums):
            coordX = info['coordX'].values[n]
            coordY = info['coordY'].values[n]
            coordZ = info['coordZ'].values[n]
            coord_CV = sitk_data.TransformPhysicalPointToIndex([coordX, coordY, coordZ])

            if self.LUNA is False:
                val = int(info['class'].values[n])
                diameter_mmX = info['diameter_mmX'].values[n]
                diameter_mmY = info['diameter_mmY'].values[n]
                diameter_mmZ = info['diameter_mmZ'].values[n]
            else:
                val = -1
                diameter_mmX = info['diameter_mm'].values[n]
                diameter_mmY = info['diameter_mm'].values[n]
                diameter_mmZ = info['diameter_mm'].values[n]

            px1 = coordX - diameter_mmX * 0.5
            px2 = coordX + diameter_mmX * 0.5
            py1 = coordY - diameter_mmY * 0.5
            py2 = coordY + diameter_mmY * 0.5
            pz1 = coordZ - diameter_mmZ * 0.5
            pz2 = coordZ + diameter_mmZ * 0.5
            coord_B1 = sitk_data.TransformPhysicalPointToIndex([px1, py1, pz1])
            coord_B2 = sitk_data.TransformPhysicalPointToIndex([px2, py2, pz2])
            bx1 = min(coord_B1[0], coord_B2[0])
            bx2 = max(coord_B1[0], coord_B2[0])
            by1 = min(coord_B1[1], coord_B2[1])
            by2 = max(coord_B1[1], coord_B2[1])
            bz1 = min(coord_B1[2], coord_B2[2])
            bz2 = max(coord_B1[2], coord_B2[2])
            width = bx2 - bx1
            height = by2 - by1
            depth = bz2 - bz1
            results.append([coord_CV[0], coord_CV[1], coord_CV[2], width, height, depth, val])
        d[self.keys[0]] = results
        return d

    

# class SaveImageGPUd(MapTransform):
#     """
#     Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

#     Args:
#         keys: keys of the corresponding items to be transformed.
#             See also: :py:class:`monai.transforms.compose.MapTransform`
#         meta_key_postfix: `key_{postfix}` was used to store the metadata in `LoadImaged`.
#             So need the key to extract metadata to save images, default is `meta_dict`.
#             The meta data is a dictionary object, if no corresponding metadata, set to `None`.
#             For example, for data with key `image`, the metadata by default is in `image_meta_dict`.
#         output_dir: output image directory.
#         output_postfix: a string appended to all output file names, default to `trans`.
#         output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
#         resample: whether to resample before saving the data array.
#             if saving PNG format image, based on the `spatial_shape` from metadata.
#             if saving NIfTI format image, based on the `original_affine` from metadata.
#         mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

#             - NIfTI files {``"bilinear"``, ``"nearest"``}
#                 Interpolation mode to calculate output values.
#                 See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
#             - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
#                 The interpolation mode.
#                 See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

#         padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

#             - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
#                 Padding mode for outside grid values.
#                 See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
#             - PNG files
#                 This option is ignored.

#         scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
#             [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
#             it's used for PNG format only.
#         dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
#             if None, use the data type of input data. To be compatible with other modules,
#             the output data type is always ``np.float32``.
#             it's used for NIfTI format only.
#         output_dtype: data type for saving data. Defaults to ``np.float32``.
#             it's used for NIfTI format only.
#         save_batch: whether the import image is a batch data, default to `False`.
#             usually pre-transforms run for channel first data, while post-transforms run for batch data.

#     """

#     def __init__(
#         self,
#         keys: KeysCollection,
#         meta_key_postfix: str = "meta_dict",
#         output_dir: str = "./",
#         output_postfix: str = "trans",
#         output_ext: str = ".nii.gz",
#         resample: bool = True,
#         mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
#         padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
#         scale: Optional[int] = None,
#         dtype: DtypeLike = np.float64,
#         output_dtype: DtypeLike = np.float32,
#         save_batch: bool = False,
#     ) -> None:
#         super().__init__(keys)
#         self.meta_key_postfix = meta_key_postfix
#         self._saver = SaveImage(
#             output_dir=output_dir,
#             output_postfix=output_postfix,
#             output_ext=output_ext,
#             resample=resample,
#             mode=mode,
#             padding_mode=padding_mode,
#             scale=scale,
#             dtype=dtype,
#             output_dtype=output_dtype,
#             save_batch=save_batch,
#         )

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             meta_data = d[f"{key}_{self.meta_key_postfix}"] if self.meta_key_postfix is not None else None
#             self._saver.saver.output_postfix = key
#             self._saver(img=d[key], meta_data=meta_data)
#         return d