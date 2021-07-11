#!/usr/bin/env python
# coding: utf-8
# %%
# stl mesh file format to 3d volume array data
# reference:
# stl file format introduction:
# https://all3dp.com/what-is-stl-file-format-extension-3d-printing/

# matlab:
# https://stackoverflow.com/questions/51158078/load-stl-file-in-matlab-and-convert-to-a-3d-array

# trimesh:
# https://stackoverflow.com/questions/36403383/convert-stl-2-numpy-volume-data
# https://pypi.org/project/trimesh/
# https://github.com/mikedh/trimesh
# https://github.com/mikedh/trimesh/issues/200

# vtk:
# https://blog.csdn.net/mrbaolong/article/details/106407467

# imagej
# https://imagej.net/Voxelization

# others:
# https://github.com/mattatz/unity-voxel
# https://github.com/davidstutz/mesh-voxelization

# %%
import numpy as np
import trimesh
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import SimpleITK as sitk
from os import path, makedirs
import os
import sys
import subprocess
import json
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from monai.mesh import utils 


# %%
def check_closed(polyData):
    """
    check if a poly data is closed.
    """
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOn()
    featureEdges.NonManifoldEdgesOn()
    featureEdges.SetInputData(polyData)
    featureEdges.Update()
    return not featureEdges.GetOutput().GetNumberOfCells() > 0


def resample_to_image(src_img, dst_img, interpolate=sitk.sitkNearestNeighbor):
    """
    Resample src_img according to dst_img
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dst_img)
    resampler.SetInterpolator(interpolate)
    return resampler.Execute(src_img)


def voxelize_stl(mesh, pitch):
    """
    Voxelize a stl mesh to a voxel array by using trimesh
    """
    volume = mesh.voxelized(pitch=pitch)
    mat = volume.matrix  # matrix of boolean
    # we need fill the holes since only the boundary is filled
    mat = ndimage.morphology.binary_fill_holes(mat)

    # transpose x,y,z order to z,y,x order, we get this by test.
    mat = np.swapaxes(mat, 0, -1)

    # set meta data
    image = sitk.GetImageFromArray(mat.astype(np.uint8))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    image.SetSpacing([pitch] * 3)
    image.SetOrigin(mesh.bounds[0])
    return image


def voxelize_stl_to_ct(stl_mesh, ct_img, pitch=None):
    """
    Voxelize a stl mesh(trimesh) and resample&align to its original CT image.
    That means the result image should have the same shape/direction/size/origin/spaceing with the CT
    """
    if pitch is None:
        pitch = np.min(ct_img.GetSpacing()) / 2.0
        vmin_extent = stl_mesh.extents.min()
        if vmin_extent / pitch < 10:
            pitch = vmin_extent / 10
            print(f'descrease pitch: {pitch}')

        if vmin_extent / pitch > 512:
            pitch = vmin_extent / 512
            print(f'increase pitch: {pitch}')

    img = voxelize_stl(stl_mesh, pitch)
    img = resample_to_image(img, ct_img)
    return img


def voxelize_poly(stl, spacing):
    """
    voxelize a vtkPolyData
    """
    if stl.GetNumberOfCells() < 1 or stl.GetNumberOfPoints() < 1:
        return None

    if isinstance(spacing, (int, float, np.number)):
        spacing = [spacing] * 3

    bds = [0] * 6
    stl.GetBounds(bds)
    bds = np.array(bds)
    dims = np.ceil((bds[1::2] - bds[0::2]) / spacing).astype(np.int)
    image = vtk.vtkImageData()
    image.SetOrigin(bds[::2])
    image.SetSpacing(spacing)
    image.SetDimensions(dims)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)
    dataToStencil = vtk.vtkPolyDataToImageStencil()
    dataToStencil.SetInputData(stl)
    dataToStencil.SetOutputSpacing(image.GetSpacing())
    dataToStencil.SetOutputOrigin(image.GetOrigin())

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(dataToStencil.GetOutputPort())
    stencil.ReverseStencilOn()
    stencil.SetBackgroundValue(0)

    stencil.Update()
    res = stencil.GetOutput()
    res = dsa.WrapDataObject(res)
    res.PointData['ImageScalars'][:] = np.invert(
        res.PointData['ImageScalars'].astype(np.bool))
    return res.VTKObject


def voxelize_mesh_to_ct(stl_mesh, ct_img, pitch):
    """
    Voxelize a stl mesh(poly data) and resample&align to its original CT image.
    That means the result image should have the same shape/direction/size/origin/spaceing with the CT
    """
    img = voxelize_poly(stl_mesh, pitch)
    img = utils.vtk_to_itk(img)
    img = utils.downsample_mask(img, ct_img.GetSpacing(), order=1)
    img = resample_to_image(img, ct_img)
    return img

