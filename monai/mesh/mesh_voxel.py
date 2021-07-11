import SimpleITK as sitk
import vtk
import numpy as np
from .utils import (itk_to_vtk, numpy_to_vtk_image, vtk_image_to_numpy)


def vtk_mesh_pipeline(data, stl_path: str, option: dict = {}):
    """
    3D reconstruction by vtk image.
    ref: https://lorensen.github.io/VTKExamples/site/Python/DataManipulation/MeshLabelImageColor/
    """
    if isinstance(data, sitk.Image):
        data = itk_to_vtk(data)
    elif isinstance(data, np.ndarray):
        data = numpy_to_vtk_image(data)

    # pad zero to avoid hole.
    arr = np.pad(vtk_image_to_numpy(data), [(1, 1)] * 3).astype(np.float32)

    arr[arr == 0] = -1

    # reset spacing, origin and direction since padding
    offset = [1] * 3
    dir_mat = data.GetDirectionMatrix()
    dir_mat.MultiplyPoint(data.GetSpacing(), offset)
    origin = np.array(data.GetOrigin()) - offset
    data = numpy_to_vtk_image(arr, data.GetSpacing(), origin, dir_mat)

    if not isinstance(data, (vtk.vtkDataObject, )):
        raise ('Unsupported data type!')

    if 'mask' in option and isinstance(option['mask'], vtk.vtkAlgorithm):
        option['mask'].SetInputData(data)
        option['mask'].Update()
        data = option['mask'].GetOutput()
    else:
        gauss_smooth = vtk.vtkImageGaussianSmooth()
        gauss_smooth.SetInputData(data)
        gauss_smooth.SetStandardDeviation(0.8, 0.8, 0.8)
        gauss_smooth.SetRadiusFactors(1.5, 1.5, 1.5)
        gauss_smooth.Update()
        data = gauss_smooth.GetOutput()

    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(data)
    marching_cubes.ReleaseDataFlagOn()
    marching_cubes.ComputeScalarsOff()
    marching_cubes.ComputeGradientsOff()
    marching_cubes.SetNumberOfContours(1)
    marching_cubes.SetValue(0, 0)
    marching_cubes.Update()
    data = marching_cubes.GetOutput()

    if 'decimate' in option and isinstance(option['decimate'],
                                           vtk.vtkAlgorithm):
        option['decimate'].SetInputData(data)
        option['decimate'].Update()
        data = option['decimate'].GetOutput()
    else:
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(data)
        decimate.SetTargetReduction(0.6)
        decimate.PreserveTopologyOn()
        decimate.Update()
        data = decimate.GetOutput()

    if 'smooth' in option and isinstance(option['smooth'], vtk.vtkAlgorithm):
        option['smooth'].SetInputData(data)
        option['smooth'].Update()
        data = option['smooth'].GetOutput()
    else:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(data)
        smoother.ReleaseDataFlagOn()
        smoother.SetNumberOfIterations(20)
        smoother.SetRelaxationFactor(0.01)
        smoother.SetFeatureAngle(45)
        smoother.SetConvergence(0)
        smoother.Update()
        data = smoother.GetOutput()

    if stl_path:
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(data)
        writer.SetFileTypeToBinary()
        writer.SetFileName(stl_path)
        writer.Write()
    return data


# -
def mesh_mask(data,
              stl_path: str,
              index: int = 1,
              smooth: bool = True,
              reduction: float = 0.8):
    """
    3D reconstruction by vtk image.
    ref: https://lorensen.github.io/VTKExamples/site/Python/DataManipulation/MeshLabelImageColor/
    """
    if isinstance(data, sitk.Image):
        data = itk_to_vtk(data)
    elif isinstance(data, np.ndarray):
        data = numpy_to_vtk_image(data)

    # pad zero to avoid hole.
    arr = np.pad(vtk_image_to_numpy(data), [(1, 1)] * 3)
    data = numpy_to_vtk_image(arr, data.GetSpacing(),
                              np.array(data.GetOrigin()) - data.GetSpacing(),
                              data.GetDirectionMatrix())

    if not isinstance(data, (vtk.vtkDataObject, )):
        raise ('Unsupported data type!')

    contour = vtk.vtkDiscreteMarchingCubes()  # For label images.
    contour.SetInputData(data)

    contour.SetValue(0, index)
    contour.Update()

    res = contour
    if smooth:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(
            30)  # This has little effect on the error!
        # smoother.BoundarySmoothingOn()
        # smoother.BoundarySmoothingOff()

        # smoother.FeatureEdgeSmoothingOff()
        # smoother.SetFeatureAngle(120.0)
        # smoother.SetPassBand(.001)        # This increases the error a lot!
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.GenerateErrorScalarsOn()
        # smoother.GenerateErrorVectorsOn()
        smoother.Update()

        res = smoother

    if reduction is not None:
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(res.GetOutput())
        decimate.SetTargetReduction(reduction)
        decimate.PreserveTopologyOn()
        decimate.Update()

        res = decimate

    if stl_path:
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(res.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(stl_path)
        writer.Write()

    return res.GetOutput()


if __name__ == '__main__':
    import pyvista as pv
    from IPython.display import display
    fname = '~/nnUNet/data/nnUNet_raw_data/Task002_BronchusPulmonaryVeinArtery/labelsTr/case_00000.nii.gz'
    label_itk = sitk.ReadImage(fname)
    label_arr = sitk.GetArrayFromImage(label_itk)
    one = mesh_mask(label_itk)
    two = mesh_mask(label_arr)
    p = pv.Plotter()
    p.add_mesh(one)
    display(p.show())
    p = pv.Plotter()
    p.add_mesh(two)
    display(p.show())
