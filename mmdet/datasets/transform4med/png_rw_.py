# coding=utf-8
# @Time	  : 2019-01-14 16:16
# @Author   : Monolith
# @FileName : png_rw.py

import numpy as np
import os, cv2


##窗宽窗位调整
def adjust_ww_wl(image, ww = 250, wc = 250, is_uint8 = True):
    """
    调整图像得窗宽窗位
    :param image: 3D图像
    :param ww: 窗宽
    :param wl: 窗位
    :return: 调整窗宽窗位后的图像
    """
    min_hu = wc - (ww/2)
    max_hu = wc + (ww/2)
    new_image = np.clip(image, min_hu, max_hu)#np.copy(image)
    if is_uint8:
        new_image -= min_hu
        new_image = np.array(new_image / ww * 255., dtype = np.uint8)

    return new_image


def _dcm2uint16Param(img_min, img_max, upper_limit=2 ** 16 - 16):
    """
    获取图像到png uint16线性变换的参数
    png 16位的像素值只可能是0~2^16，原始图像的可能范围是(-1000,1000)
    线性变换如下
    y  = ax + b
    :param img_min: 图像最小值
    :param img_max: 图像最大值
    :param upper_limit: 变换目标的上限
    :return: 线性变换的参数
    """

    a = upper_limit / (img_max - img_min)
    b = upper_limit * img_min / (img_min - img_max)
    return a, b


def _image_masks_in_png(png_array, label_color='red'):
    """
    从png中得到image和mask 通道
    :param png_array: 三通到png矩阵
    :param label_color: mask存储的颜色，用于解析判断mask所在的通道
    :return: image通道 和 mask通道
    """
    mask_on_image = np.zeros(png_array.shape[:2], dtype=np.int16)
    info_on_image = np.copy(mask_on_image)
    # channel_list = []
    if label_color in ['red', 'yellow']:
        mask_on_image = png_array[..., 0]
        info_on_image = png_array[..., 2]
    elif label_color == ['green', 'cyan']:
        mask_on_image = png_array[..., 2]
        info_on_image = png_array[..., 1]
    elif label_color == ['blue', 'megenta']:
        mask_on_image = png_array[..., 1]
        info_on_image = png_array[..., 0]
    else:
        print('label color can only be red, green, blue, yellow, cyan, megenta')
    # mask_on_image = np.clip(mask_on_image, label_start, label_start + num_label)
    # mask_channel = mask_on_image - label_start
    # mask_channel = np.array(mask_channel, dtype=np.int8)

    return info_on_image, mask_on_image


def _prepare_image_channels(info_channel, image_channel, mask_channel, label_color='red'):
    """
    将原始图像、mask和信息按一定顺序合并到一个三通道矩阵中
    :param info_channel: 包含图像信息的通道 如spacing, SOP, IPP
    :param image_channel: 原图通道
    :param mask_channel: mask通道
    :param label_color: mask应该呈现的颜色
    :return: 三通道矩阵
    """
    # 三个通道都包含原图信息

    # 信息通道
    info_on_image_channel = np.copy(image_channel)
    info_on_image_channel[:4, :30] = info_channel[:4, :30]

    # mask通道
    mask_on_image_channel = np.copy(image_channel)
    mask_on_image_channel[mask_channel > 0] = mask_channel[mask_channel > 0]
    mask_on_image_channel[:4, :30] = info_channel[:4, :30]

    #  一通道blue, 二通道green, 三通道red
    if label_color == 'red':
        channel_list = [mask_on_image_channel, mask_on_image_channel, info_on_image_channel]
    elif label_color == 'green':
        channel_list = [mask_on_image_channel, info_on_image_channel, mask_on_image_channel]
    elif label_color == 'blue':
        channel_list = [info_on_image_channel, mask_on_image_channel, mask_on_image_channel]
    elif label_color == 'yellow':  # red + green
        channel_list = [mask_on_image_channel, info_on_image_channel, info_on_image_channel]
    elif label_color == 'cyan':  # blue + green
        channel_list = [info_on_image_channel, info_on_image_channel, mask_on_image_channel]
    elif label_color == 'megenta':  # blue + red
        channel_list = [info_on_image_channel, mask_on_image_channel, info_on_image_channel]
    else:
        print('label color can only be red, green, blue, yellow, cyan, megenta')
        raise ValueError('label_color is invalid')
    png_array = np.stack(channel_list, axis=2)
    png_array_16 = np.array(png_array, dtype=np.uint16)

    return png_array_16


def write_array2png_dl(img_2d, save_dir, file_name, shift_quant = 32768):
    """ based on deep lesion convention
        the img_2d should be in int16
    """
    img_2d += shift_quant
    png_array_16 = np.array(img_2d, dtype=np.uint16)
    cv2.imwrite(os.path.join(save_dir, file_name), png_array_16)  # b, g, r


def read_png2array_dl(png_path, shift_quant = 32768):
    png_array = cv2.imread(png_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return png_array - shift_quant

def write_array2png(img_2d, save_dir, file_name,
                    spacing, mask_2d = None, 
                    label_start_value=2 ** 13, label_color='red'):
    """
    将单张图及其label一起存储到png，方便筛查
    png 16位的像素值只可能是0~2^16，因此需要对图片的像素值以及spacing进行线性变换，以便显示，并保留数值精度
    :param img_2d: 原始图像
    :param mask_2d: label图像
    :param save_dir: 存储路径(只接受英文名)
    :param file_name: 文件名，以png结尾(只接受英文名)
    :param spacing: 图像的spacing，三个元素，z,y,x
    :param label_start_value: 存储成目标格式时的取值范围
    :return:
    """
    mask_2d = np.zeros_like(img_2d) if mask_2d is None else mask_2d

    img_min = np.amin(img_2d)
    img_max = np.amax(img_2d)
    if img_max > 1e4:
        print('there are extreme Hu value in the volume, %s' % img_max)
        img_max = 5e3
        img_2d = np.clip(img_2d, img_min, img_max)
    a, b = _dcm2uint16Param(img_min, img_max)
    img_channel = np.array((img_2d) * a + b, dtype=np.int32)
    # 建立空通道，存放变换参数
    info_channel = np.zeros(img_channel.shape, dtype=np.uint16)
    # 存spacing: 转换成正整数，保存精度，uint16
    info_channel[0, :3] = np.array(spacing) * 1e4
    info_channel[0, 3] = 1e4
    # 存HU值上下限
    min2posi = np.abs(img_min)  ##HU值的下限，转换成正整数
    info_channel[1, :3] = img_min + min2posi, img_max + min2posi, min2posi

    # 对mask进行数值变换 大于0的值 加上upper_limit 从而更好的显示，适用于有很多label的mask
    # 只需存储upper_limit就可以了

    unique_label = np.unique(mask_2d)[1:]
    # 存label值上限
    info_channel[2, 0] = label_start_value  # HU值的下限，转换成正整数

    if unique_label.any():
        # label_param = upper_limit/label_max_val
        mask_channel = np.zeros(mask_2d.shape, dtype=np.uint16)
        for k in range(unique_label.size):
            info_channel[2, k + 1] = unique_label[k]
            mask_temp = np.array(mask_2d == unique_label[k], dtype=np.uint16) * (label_start_value + unique_label[k])
            mask_channel = np.amax(np.stack([mask_channel, mask_temp], axis=0), axis=0)
    else:
        mask_channel = mask_2d

    png_array_16 = _prepare_image_channels(info_channel,
                                           img_channel,
                                           mask_channel,
                                           label_color)
    # print(os.path.join(save_dir, file_name))
    # 序号、医院、患者ID、层号、label号
    save_path = os.path.join(save_dir, file_name)
    cv2.imwrite(save_path, png_array_16)  # b, g, r
    return save_path


def read_png2array(png_path, label_color='red'):  # , label_mode='fill'
    """
    读取png int16 图片，获取原始图像和label
    :param png_path:  png所在路径
    :param label_color: label在png中的颜色
    :return: 原始图像，label图像，spacing
    """
    
    png_array = cv2.imread(str(png_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # 第一个通道全是空，第二个通道是img，第三个通道是mask
    # 第一个通道存放了原图+基本信息
    info_on_image_channel, mask_on_image_channel = _image_masks_in_png(png_array, label_color)
    # from lib.utlis.visualization import plotOneImage
    # plotOneImage(png_array[...,2])
    # spacing存在了第二个通道首行的三前列
    spacing = np.array(info_on_image_channel[0, :3], np.float) / info_on_image_channel[0, 3]
    # 取hu的映射关系
    img_min, img_max = np.array(info_on_image_channel[1, :2], np.int32) - info_on_image_channel[1, 2]
    if img_min == img_max:
        print('\t@@min==max: ', png_path)
    # 取label的映射关系
    label_start = info_on_image_channel[2, 0]
    label_values = np.array([a for a in info_on_image_channel[2, 1:20] if a != 0])
    # 将存基本信息的像素点全部还原成0
    info_on_image_channel[:3, :20] = 0

    # 基于图像最大最小值得到映射参数
    a, b = _dcm2uint16Param(img_min, img_max)

    # 还原图像
    img_channel = np.array((info_on_image_channel - b) / a, dtype=np.int16)

    # 还原label
    # num_label = len(png_path.split(os.sep)[-1].split('.')) -5
    mask_multi_label = np.zeros_like(img_channel, dtype=np.uint8)
    # from lib.utlis.visualization import plot2Image, plotOneImage
    if label_values.size > 0:
        for k in label_values:
            # print(k, label_start + k)
            mask_k = np.array(mask_on_image_channel == label_start + k, dtype=np.uint8)
            # remove salty spot in mask
            mask_k = cv2.medianBlur(mask_k, 3)
            mask_multi_label += mask_k * k

    # plot2Image(mask_on_image_channel, mask_multi_label)
    # 去噪
    # mask_multi_label = remove_mask_noises(mask_multi_label, min_size=5)

    return img_channel, mask_multi_label, spacing



class IO4Png():

    SPACEING_SCALER = 1e4

    def __init__(self, save_dir = None, spacing=None, 
                max_value_allowed = 5e3, 
                label_start_value=2 ** 13, 
                label_color='red', 
                img_min = None, 
                img_max = None,
                verbose = False
                ):
        self.save_dir = save_dir
        self.spacing = spacing
        self.max_value_allowed = max_value_allowed
        self.label_start_value = label_start_value
        self.label_color = label_color
        self.img_min = img_min
        self.img_max = img_max
        self.verbose = verbose
    

    def write(self, img_2d, file_name, mask_2d = None, is_3_channel = False):
        """
        将单张图及其label一起存储到png，方便筛查
        png 16位的像素值只可能是0~2^16，因此需要对图片的像素值以及spacing进行线性变换，以便显示，并保留数值精度
        :param img_2d: 原始图像
        :param mask_2d: label图像
        :param save_dir: 存储路径(只接受英文名)
        :param file_name: 文件名，以png结尾(只接受英文名)
        :param spacing: 图像的spacing，三个元素，z,y,x
        :param label_start_value: 存储成目标格式时的取值范围
        :return:
        """
        mask_2d = np.zeros_like(img_2d) if mask_2d is None else mask_2d

        img_min = np.amin(img_2d) if self.img_min is None else self.img_min
        img_max = np.amax(img_2d) if self.img_max is None else self.img_max
        if self.verbose: print('Img extreme', img_min, img_max)
        if img_max > self.max_value_allowed:
            print('there are extreme Hu value in the volume, %s' % img_max)
            img_2d = np.clip(img_2d, img_min, self.max_value_allowed)
        a, b = _dcm2uint16Param(img_min, img_max)
        img_channel = np.array((img_2d) * a + b, dtype=np.int32)
        # 建立空通道，存放变换参数
        info_channel = img_channel.astype(np.uint16)#np.zeros(img_channel.shape, dtype=np.uint16)
        # 存spacing: 转换成正整数，保存精度，uint16
        info_channel[0, :3] = np.array(self.spacing) * self.SPACEING_SCALER
        info_channel[0, 3] = self.SPACEING_SCALER
        # 存HU值上下限
        min2posi = np.abs(img_min)  ##HU值的下限，转换成正整数
        info_channel[1, :3] = img_min + min2posi, img_max + min2posi, min2posi
        if self.verbose: print('Store extreme', info_channel[1, :3])
        # 对mask进行数值变换 大于0的值 加上upper_limit 从而更好的显示，适用于有很多label的mask
        # 只需存储upper_limit就可以了

        unique_label = np.unique(mask_2d)[1:]
        # 存label值上限
        info_channel[2, 0] = self.label_start_value  # HU值的下限，转换成正整数

        if unique_label.any():
            # label_param = upper_limit/label_max_val
            mask_channel = np.zeros(mask_2d.shape, dtype=np.uint16)
            for k in range(unique_label.size):
                info_channel[2, k + 1] = unique_label[k]
                mask_temp = np.array(mask_2d == unique_label[k], dtype=np.uint16) * (self.label_start_value + unique_label[k])
                mask_channel = np.amax(np.stack([mask_channel, mask_temp], axis=0), axis=0)
        else:
            mask_channel = mask_2d

        if is_3_channel:
            png_array_16 = _prepare_image_channels(info_channel,
                                                img_channel,
                                                mask_channel,
                                                self.label_color)
        else:
            png_array_16 = info_channel
        # print(os.path.join(save_dir, file_name))
        # 序号、医院、患者ID、层号、label号
        save_path = os.path.join(self.save_dir, file_name)
        cv2.imwrite(save_path, png_array_16)  # b, g, r
        return save_path

    def read(self, png_path, dtype = np.int16):

        """
        读取png int16 图片，获取原始图像和label
        :param png_path:  png所在路径
        :param label_color: label在png中的颜色
        :return: 原始图像，label图像，spacing
        """
        assert os.path.exists(png_path), f'{png_path} not exist'
        png_array = cv2.imread(str(png_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # 第一个通道全是空，第二个通道是img，第三个通道是mask
        # 第一个通道存放了原图+基本信息
        if len(png_array.shape) == 3:
            info_on_image_channel, mask_on_image_channel = _image_masks_in_png(png_array, self.label_color)
        else:
            info_on_image_channel, mask_on_image_channel = png_array, None
        # from lib.utlis.visualization import plotOneImage
        # plotOneImage(png_array[...,2])
        # spacing存在了第二个通道首行的三前列
        spacing = np.array(info_on_image_channel[0, :3], np.float) / info_on_image_channel[0, 3]
        # 取hu的映射关系
        # img_min, img_max = np.array(info_on_image_channel[1, :2], np.int32) - info_on_image_channel[1, 2]
        if self.img_min is None:
            img_min = np.array(info_on_image_channel[1, 0], np.int32) - info_on_image_channel[1, 2]
        else:
            img_min = self.img_min
        
        if self.img_max is None:
            img_max = np.array(info_on_image_channel[1, 1], np.int32) - info_on_image_channel[1, 2]
        else:
            img_max = self.img_max

        if self.verbose: print('Check extreme', img_min, img_max)
        # 取label的映射关系
        label_start = info_on_image_channel[2, 0]
        label_values = np.array([a for a in info_on_image_channel[2, 1:20] if a != 0])
        # 将存基本信息的像素点全部还原成0
        info_on_image_channel[:3, :20] = 0

        # 基于图像最大最小值得到映射参数
        a, b = _dcm2uint16Param(img_min, img_max)

        # 还原图像
        img_channel = np.array((info_on_image_channel - b) / a, dtype=dtype)

        # 还原label
        # num_label = len(png_path.split(os.sep)[-1].split('.')) -5
        mask_multi_label = np.zeros_like(img_channel, dtype=np.uint8)
        # from lib.utlis.visualization import plot2Image, plotOneImage
        if mask_on_image_channel is not None and label_values.size > 0:
            for k in label_values:
                # print(k, label_start + k)
                mask_k = np.array(mask_on_image_channel == label_start + k, dtype=np.uint8)
                mask_k = cv2.medianBlur(mask_k, 3)
                mask_multi_label += mask_k * k

        # plot2Image(mask_on_image_channel, mask_multi_label)
        # 去噪
        # mask_multi_label = remove_mask_noises(mask_multi_label, min_size=5)

        return img_channel, mask_multi_label, spacing

def save_img_label2png(info_dict, image_3d, label_mask, store_path, p_ind,
                       visual_window, label_color: str, save_mode='contour',
                       label_value=2 ** 16 - 100):  # , visual_path
    """
    将原始图像和预测mask存储为png
    :param info_dict:
    :param image_3d:
    :param label_mask:
    :param store_path:
    :param p_ind:
    :param save_mode:
    :param window_para: 原始图像的取值范围，也就是窗宽窗位
    :return:
    """

    image_3d = adjust_ww_wl(image_3d, ww=visual_window[0], wl=visual_window[1])
    # 生成这里数据的文件夹名称
    hosp = 'hosp_x' if info_dict.hospital in ['', None] else info_dict.hospital

    print('save prediction mask to png for visual check')
    # 存储原始图像和标注成png格式
    for s in range(image_3d.shape[0]):
        # print(s)
        vol_z_ind = str(s)
        mask_2d = label_mask[s]
        label_values = np.unique(mask_2d)[1:]
        if label_values.size > 0:
            if save_mode == 'contour':
                mask_2d = contour_mask3labels(mask_2d)

            slice_label = [info_dict.organ_names[int(v) - 1] for v in label_values
                           if v in range(len(info_dict.organ_names) + 1)]
        else:
            slice_label = ['nan']
        file_name = '.'.join([str(p_ind), hosp[:8], info_dict.pid[-10:],
                              vol_z_ind, '.'.join(slice_label), 'png'])

        write_array2png(image_3d[s], mask_2d, store_path, file_name, info_dict.spacing_list,
                        label_color=label_color, label_start_value=label_value)


def image_label2png(ind, image_2d, label_2d,
                    pid, spacing_list, body_part, hosp,
                    pat_root, roi_include,  **kwargs):
    # spacing_list = kwargs['spacing_list']
    # pat_root = kwargs['pat_root']
    # roi_include = kwargs['roi_include']
    # hosp = kwargs['hosp']
    # pid = kwargs['pid']
    # body_part = kwargs['body_part']

    vol_z_ind = '%03d'%int(ind)
    label_value = list(np.unique(label_2d))
    label_value.pop(0)

    if len(label_value) > 0:
        slice_label = {roi_include[a - 1] for a in label_value}
    else:
        slice_label = ['nan']
    file_name = '.'.join([hosp[:8], pid[-10:], body_part,
                          vol_z_ind, '.'.join(list(slice_label)), 'png'])

    # label_2d = remove_mask_noises(label_2d, min_size=5)
    # print('   mask unique value %s' % list(np.unique(label_2d)))

    write_array2png(image_2d, label_2d, pat_root, file_name, spacing_list,
                    label_start_value=2 ** 13, label_color='red')

if __name__ == '__main__':

    data_root = r'/media/dejun/holder/lith/chest_organs/train_png_good/173.chest.somehosp.E0074'
    img_name = r'img_label_49.npy'
    png_name = 'test.png'

    img_label_path = os.path.join(data_root, img_name)
    img_label = np.load(img_label_path)
    spacing = [2.5, 0.9754, 0.9754]
    img_2d = img_label[0]
    mask_2d = img_label[1]
    write_array2png(img_2d, mask_2d, data_root, png_name,
                    spacing, label_start_value=2 ** 13, label_color='red')

    x, y, spacing = read_png2array(os.path.join(data_root, png_name), label_color='red')
    from utils.visualization import plot2Image

    plot2Image(mask_2d, y)
    print(np.unique(y))

    # png_file_path = os.path.join(data_root, file_name)
    # img_channel, mask_channel, spacing = read_png2array(png_file_path)
    # print(np.amax(mask_channel))
    # # img_3mm.append(img_channel)
    # # mask_3mm.append(mask_channel)
    # plt.figure()
    # plt.imshow(mask_channel)
    # plt.show()

    # img_names_unorder = os.listdir(pat_path)
    # slice_ind = np.argsort([int(s.split('.')[3]) for s in img_names_unorder])
    # slice_paths = [os.path.join(pat_path, img_names_unorder[f]) for f in slice_ind[::-1]]
    # x_vol = []
    # y_vol = []
    # for f in slice_paths:
    #     x, y, spacing = read_png2array(f, label_color='red')
    #     from lkm_lib.utlis.visualization import plotOneImage
    #
    #     # plotOneImage(x, cmap= 'gray')
    #     x_vol.append(x)
    #     y_vol.append(y)
    # x_vol_img = np.stack(x_vol, axis=0)
    # x_vol_img = np.clip(x_vol_img, -1000, 1000)
    # from lkm_lib.utlis.visualization import plotOneImage
    #
    # plotOneImage(x_vol_img[:, 250, :], cmap='gray')
