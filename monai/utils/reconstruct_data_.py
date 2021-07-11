import numpy as np
import SimpleITK as sitk
from typing import Optional, Union
from monai.transforms.spatial.array_ import Affine_

from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
)


class Reconstruct_data:
    def __init__(self,
                 image: sitk.Image,
                 mask_crop: Union[Optional[str], np.ndarray],
                 target_spacing: np.ndarray = np.ones(3),
                 target_orientation: np.ndarray = np.eye(3),
                 ):
        self.image = image
        self.mask_crop = [0, 0, 0, self.image.GetSize()[0]-1, self.image.GetSize()[1]-1, self.image.GetSize()[2]-1]
        if mask_crop is not None:
            self.mask_crop = mask_crop
        self.target_spacing = target_spacing
        self.target_orientation = target_orientation
        self.origin_last = (0, 0, 0)

        # construct affine matrix
        self.affine_matrix_vorigin_to_final = self.calculate_affine_matrix()


    def calculate_affine_matrix(self):
        '''
        Volume0 -> Volume1 is cropped by lung mask
        then we have:
        |Volume1| + crop = |Volume0|
        R1*S1*|Volume0| + Origin0 = R2*S2*|Volume2| + Origin2
        so we have:
        R1*S1*|Volume0| + Origin0 - Origin2 = |Volume2|
        R1*S1*|Volume1+crop| + Origin0 - Origin2 = |Volume2|

        where,
        R1 is direction of input image
        S1 is spacing of input image
        R2, S2 are identify matrix
        so we only need to calculate Origin0
        1. get 8 corners of Volume1
        2. transform 8 corners by function: R1*S1*|Volume0| + Origin0 (re-spacing and re-orientation)
        3. calculate min and max bbox of transformed 8 corners. Origin2=-1.0*min(trans_corners)

        then we have:
        R1*S1*|Volume0| + Origin0 - Origin2 = |Volume2|
        R1*S1*|Volume1| + R1*S1*crop + Origin0 - Origin2 = |Volume2|
        Origin1 = R1*S1*crop + Origin0

        then we have:
        R1*S1*|Volume0| + Origin0 - Origin2 = |Volume2|
        R1*S1*|Volume1| + R1*S1*crop + Origin0 - Origin2 = |Volume2|

        around image center as rotation center (disable output):
        R1*S1*|Volume0 - center0| + R1*S1*center0 + Origin0 - Origin2 - center2 = |Volume2 - center2|
        R1*S1*|Volume1 - center1| + R1*S1*center1 + R1*S1*crop + Origin0 - Origin2 -center2 = |Volume2 - center2|
        '''
        image = self.image
        mask_crop = self.mask_crop
        S1 = np.diag(image.GetSpacing())
        R1 = np.asmatrix(image.GetDirection()).reshape(3, 3)
        Origin0 = np.asmatrix(image.GetOrigin()).T
        # crop = np.asmatrix([mask_crop[0], mask_crop[1], mask_crop[2]]).T
        volume1_x0_y0_z0 = [mask_crop[0], mask_crop[1], mask_crop[2]]
        volume1_x1_y0_z0 = [mask_crop[3], mask_crop[1], mask_crop[2]]
        volume1_x0_y1_z0 = [mask_crop[0], mask_crop[4], mask_crop[2]]
        volume1_x1_y1_z0 = [mask_crop[3], mask_crop[4], mask_crop[2]]
        volume1_x0_y0_z1 = [mask_crop[0], mask_crop[1], mask_crop[5]]
        volume1_x1_y0_z1 = [mask_crop[3], mask_crop[1], mask_crop[5]]
        volume1_x0_y1_z1 = [mask_crop[0], mask_crop[4], mask_crop[5]]
        volume1_x1_y1_z1 = [mask_crop[3], mask_crop[4], mask_crop[5]]
        volume1_corner = [
            volume1_x0_y0_z0,
            volume1_x1_y0_z0,
            volume1_x0_y1_z0,
            volume1_x1_y1_z0,
            volume1_x0_y0_z1,
            volume1_x1_y0_z1,
            volume1_x0_y1_z1,
            volume1_x1_y1_z1,
        ]
        volume1_bbox_corner_mm = []
        for v in volume1_corner:
            trans_v = R1 * S1 * np.asmatrix(v).T + Origin0
            volume1_bbox_corner_mm.append(trans_v)

        minx, miny, minz = volume1_bbox_corner_mm[0][0, 0], volume1_bbox_corner_mm[0][1, 0], volume1_bbox_corner_mm[0][2, 0]
        maxx, maxy, maxz = volume1_bbox_corner_mm[0][0, 0], volume1_bbox_corner_mm[0][1, 0], volume1_bbox_corner_mm[0][2, 0]
        for coord in volume1_bbox_corner_mm:
            minx = min(minx, coord[0, 0])
            miny = min(miny, coord[1, 0])
            minz = min(minz, coord[2, 0])
            maxx = max(maxx, coord[0, 0])
            maxy = max(maxy, coord[1, 0])
            maxz = max(maxz, coord[2, 0])
        Origin2 = 1.0 * np.asmatrix([minx, miny, minz]).T  # note here should multiply by -1.0
        self.origin_last = (minx, miny, minz)
        # then we have , R1, S1, crop, T, R2 and S2 are identify matrix
        affine_matrix_volume0_to_volume2 = np.eye(4)
        affine_matrix_volume0_to_volume2[0:3, 0:3] = R1 * S1
        affine_matrix_volume0_to_volume2[0:3, 3:4] = Origin0 - Origin2

        return affine_matrix_volume0_to_volume2

    def get_affine_matrix_R2F(self):
        return self.affine_matrix_vorigin_to_final

    def get_affine_matrix_F2R(self):
        return np.linalg.inv(self.affine_matrix_vorigin_to_final)

    def reconstruct_image(self,
                          original_image: sitk.Image,
                          mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
                          padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
                          ):
        '''
        image: Dimension is [C, D, H, W]
        '''

        trans = Affine_(
            self.get_affine_matrix_R2F(),
            spatial_size=None,
            mode=mode,
            padding_mode=padding_mode)

        image_arr = sitk.GetArrayFromImage(original_image)
        image_arr = image_arr[np.newaxis, :]
        image_arr = np.transpose(image_arr, [0, 3, 2, 1])
        image_arr = trans(img=image_arr)
        image_arr = np.transpose(image_arr, [0, 3, 2, 1])

        new_spacing = tuple(self.target_spacing.tolist())
        new_direction = tuple(self.target_orientation.flatten().tolist())
        new_image_data = sitk.GetImageFromArray(image_arr[0])
        new_image_data.SetSpacing(new_spacing)
        new_image_data.SetDirection(new_direction)
        new_image_data.SetOrigin(self.origin_last)
        return new_image_data

    def reverse_image(self,
                      corrected_image: sitk.Image,
                      mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
                      padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
                      ):
        '''
        image: Dimension is [C, D, H, W]
        '''

        trans = Affine_(
            self.get_affine_matrix_F2R(),
            spatial_size=self.image.GetSize(),
            mode=mode,
            padding_mode=padding_mode)

        image_arr = sitk.GetArrayFromImage(corrected_image)
        image_arr = image_arr[np.newaxis, :]
        image_arr = np.transpose(image_arr, [0, 3, 2, 1])
        image_arr = trans(img=image_arr, spatial_size=np.asarray(self.image.GetSize()))
        image_arr = np.transpose(image_arr, [0, 3, 2, 1])

        new_image_data = sitk.GetImageFromArray(image_arr[0])
        new_image_data.CopyInformation(self.image)
        return new_image_data

    def __call__(self,
                 image: sitk.Image,
                 mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
                 padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
                 step: str = 'reconstruct', # 'reverse'
                 ):
        if step == 'reconstruct':
            return self.reconstruct_image(image, mode, padding_mode)
        elif step == 'reverse':
            return self.reverse_image(image, mode, padding_mode)