import SimpleITK as sitk
import vtk
import numpy as np
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
from scipy.ndimage import map_coordinates, distance_transform_edt


# +
def vtk_image_to_numpy(img):
    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    arr = arr.reshape(img.GetDimensions()[::-1])
    return arr


def numpy_to_vtk_image(arr, spacing=None, origin=None, direction=None):
    """
    convert from numpy.ndarray image to vtk image.
    """
    image = vtk.vtkImageData()
    if origin is not None: image.SetOrigin(origin)
    if spacing is not None: image.SetSpacing(spacing)
    if direction is not None: image.SetDirectionMatrix(direction)

    image.SetDimensions(arr.shape[::-1])
    image.AllocateScalars(numpy_support.get_vtk_array_type(arr.dtype), 1)
    image = dsa.WrapDataObject(image)
    image.PointData['ImageScalars'][:] = arr.ravel()
    return image.VTKObject


def numpy_to_itk_image(arr, spacing=None, origin=None, direction=None):
    """
    convert from numpy.ndarray image to itk image.
    """
    image = sitk.GetImageFromArray(arr)
    if origin is not None: image.SetOrigin(origin)
    if spacing is not None: image.SetSpacing(spacing)
    if direction is not None: image.SetDirection(direction)
    return image


def numpy_to_vtk_array(data):
    """
    convert from numpy.ndarray image to vtk array.
    """
    vtk_arr = numpy_support.numpy_to_vtk(
        data.ravel(),
        deep=True,
        array_type=numpy_support.get_vtk_array_type(data.dtype))
    return vtk_arr


def itk_to_vtk(img_itk):
    """
    convert from simpleitk image to vtk image.
    ref: https://itk.org/ITKExamples/src/Bridge/VtkGlue/ConvertAnitkImageTovtkImageData/Documentation.html
    ref: https://stackoverflow.com/questions/45395269/numpy-uint8-t-arrays-to-vtkimagedata
    """
    label = sitk.GetArrayFromImage(img_itk)
    vtk_arr = numpy_to_vtk_array(label)
    img_vtk = vtk.vtkImageData()
    img_vtk.SetSpacing(img_itk.GetSpacing())
    img_vtk.SetOrigin(img_itk.GetOrigin())
    img_vtk.SetDimensions(*reversed(label.shape))
    img_vtk.SetDirectionMatrix(img_itk.GetDirection())

    img_vtk.GetPointData().SetScalars(vtk_arr)
    img_vtk.Modified()
    return img_vtk


def vtk_to_itk(img_vtk):
    """
    convert from vtk to itk
    """
    res = vtk_image_to_numpy(img_vtk)
    res = sitk.GetImageFromArray(res)
    res.SetSpacing(img_vtk.GetSpacing())
    res.SetOrigin(img_vtk.GetOrigin())
    dire = img_vtk.GetDirectionMatrix()
    dire = np.array([[dire.GetElement(j, i) for i in range(3)]
                     for j in range(3)]).flatten()
    res.SetDirection(dire)
    return res


def mask_to_tp_fn_fp(truth, pred):
    """
    convert binary truth and pred to tp: 1, fn: 2, fp: 3
    """
    if isinstance(truth, sitk.Image):
        truth_arrs = sitk.GetArrayFromImage(truth)
    else:
        truth_arrs = truth
    if isinstance(pred, sitk.Image):
        pred_arrs = sitk.GetArrayFromImage(pred)
    else:
        pred_arrs = pred

    keys = np.unique(truth_arrs)[1:]
    pred_arrs = [(pred_arrs == i).astype(np.bool) for i in keys]
    truth_arrs = [(truth_arrs == i).astype(np.bool) for i in keys]

    splits = {}
    for k, t, p in zip(keys, truth_arrs, pred_arrs):
        tp = t * p
        res = np.zeros(t.shape, dtype=np.uint8)
        res[tp] = 1
        res[t ^ tp] = 2
        res[p ^ tp] = 3
        splits[k] = res
    return splits


def downsample_mask(src_img, dst_spacing, *args, **kargs):
    src_spacing = np.flip(src_img.GetSpacing())
    dst_spacing = np.flip(dst_spacing)
    src_arr = sitk.GetArrayFromImage(src_img)
    
    src_shape = np.array(src_arr.shape)
    dst_shape = np.array(src_arr.shape) * src_spacing / dst_spacing
    dst_shape = np.round(dst_shape).astype(np.uint64)
    dst_spz, dst_shape[0] = dst_shape[0], src_shape[0]
    # print('dst shape: ', dst_spz, dst_shape)
    
    z,y,x = np.mgrid[:dst_shape[0], :dst_shape[1], :dst_shape[2]]
    z = (src_shape[0] / dst_shape[0]) * (z + 0.5) - 0.5
    y = (src_shape[1] / dst_shape[1]) * (y + 0.5) - 0.5
    x = (src_shape[2] / dst_shape[2]) * (x + 0.5) - 0.5
    
    res_img = distance_transform_edt(src_arr)
    res_img = map_coordinates(res_img, np.array([z,y,x]), *args, **kargs)
    
    select = []
    for i in range(dst_spz):
        l = int(round(i*res_img.shape[0]/dst_spz))
        h = int(round((i+1)*res_img.shape[0]/dst_spz))
        select.append(np.max(res_img[l:h], axis=0))
            
    res_img = np.stack(select)
    res_img = (res_img > 0).astype(np.uint8)

    res_img = sitk.GetImageFromArray(res_img)
    res_img.SetSpacing(np.flip(dst_spacing))
    res_img.SetOrigin(src_img.GetOrigin())
    res_img.SetDirection(src_img.GetDirection())
    return res_img    