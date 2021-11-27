import numpy as np
import matplotlib.pylab as plt
import os.path as osp
import json, os, pdb
from tqdm import tqdm


def check_bbox_IOU(pred_bbox, gt_bboxes):
    """compute overlaps over intersection"""
    ixmin = np.maximum(gt_bboxes[:, 0], pred_bbox[0])
    iymin = np.maximum(gt_bboxes[:, 1], pred_bbox[1])
    ixmax = np.minimum(gt_bboxes[:, 2], pred_bbox[2])
    iymax = np.minimum(gt_bboxes[:, 3], pred_bbox[3])
    izmin = np.minimum(gt_bboxes[:, 4], pred_bbox[4])
    izmax = np.minimum(gt_bboxes[:, 5], pred_bbox[5])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    id = np.maximum(izmax - izmin + 1., 0.)
    inters = iw * ih * id

    # union
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) * (pred_bbox[5] - pred_bbox[4] + 1.) +
           (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.) *
           (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.) *
           (gt_bboxes[:, 5] - gt_bboxes[:, 4] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps


def check_bbox_inside(pred_bbox, gt_bboxes):
    x = (pred_bbox[0] + pred_bbox[2]) / 2
    y = (pred_bbox[1] + pred_bbox[3]) / 2
    z = (pred_bbox[4] + pred_bbox[5]) / 2
    inside_tag = []
    for n in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[n]
        # if len(gt_bbox) == 0:
        if np.all(gt_bbox == 0):
            inside_tag.append(0)
            continue

        gtx1 = gt_bbox[0]
        gtx2 = gt_bbox[2]
        gty1 = gt_bbox[1]
        gty2 = gt_bbox[3]
        gtz1 = gt_bbox[4]
        gtz2 = gt_bbox[5]

        # gtw = gtx2 - gtx1
        # gth = gty2 - gty1
        # gtd = gtz2 - gtz1
        # assert gtw >= 0 and gth >= 0 and gtd >= 0
        # if x > gtx1 + 0.45*gtw and x < gtx2 - 0.45*gtw and \
        #         y > gty1 + 0.45*gth and y < gty2 - 0.45*gth and \
        #         z > gtz1 + 0.45*gtd and z < gtz2 - 0.45*gtd:
        #     inside_tag.append(1)

        if gtx1 <= x <= gtx2 and gty1 <= y <= gty2 and gtz1 <= z <= gtz2:
            inside_tag.append(1)
        else:
            inside_tag.append(0)
    return inside_tag

def point2bbox_hitmap(pred_point_nx3, gt_bbox_kx2x3):

    assert pred_point_nx3.shape[-1] == 3
    assert gt_bbox_kx2x3.shape[-2:] == (2, 3)

    pred_n, gt_k = pred_point_nx3.shape[0], gt_bbox_kx2x3.shape[0]

    if gt_k == 0 or pred_n == 0:
        pred2gt_nxk = np.zeros((pred_n, gt_k)) - 1
    else:
        point2bbox_start = pred_point_nx3[:, None, :] -  gt_bbox_kx2x3[None, :, 0]  # nx1x3 - 1xkx3, nxkx3
        point2bbox_end = -1 * pred_point_nx3[:, None, :] + gt_bbox_kx2x3[None, :, 1]  # 
        point2bbox_nxkx6 = np.stack([point2bbox_start, point2bbox_end], axis = -1).reshape(pred_n, gt_k, 6)
        pred2gt_nxk = np.amin(point2bbox_nxkx6, axis = -1)  >= 0 # nxk
    return pred2gt_nxk


def check_center_inside(pred_bbox_v6, gt_nx6, is_strict = True):
    """

    Args:
        pred_center: 3 long vector, xyz
        gt_bboxes: nx6, x0, y0, z0, x1, y1, z1
    """

    pred_bbox_1x2x3 = np.array(pred_bbox_v6).reshape(-1, 2, 3)
    gt_bbox_kx2x3 = np.array(gt_nx6).reshape(-1, 2, 3)
    
    pred_point_1x3 = pred_bbox_1x2x3.mean(axis = 1)
    gt_point_kx3 = gt_bbox_kx2x3.mean(axis = 1)
    
    pred2gt_1xk = point2bbox_hitmap(pred_point_1x3, gt_bbox_kx2x3)

    if is_strict: 
        pred2gt_hitlist = pred2gt_1xk[0]
    else:
        gt2pred_kx1 = point2bbox_hitmap(gt_point_kx3, pred_bbox_1x2x3)
        pred2gt_hitlist = np.stack([pred2gt_1xk[0], gt2pred_kx1[:, 0]], axis = 1).max(axis = 1)
    
    inside_tag = [int(a) for a in pred2gt_hitlist]
    # pdb.set_trace()
    return inside_tag


def calculate_FROC_by_center(gt_by_case, pred_by_case, luna_output_format=True, plt_figure=False):
    """
    Args:
        gt_boxes: list[], case level, roi level, bbox (6)
        pred_centers: list[], case level, roi level, bbox (6)
    """
    # pdb.set_trace()
    # gt_boxes = np.array(gt_boxes, dtype = int)
    # pred_centers = np.array(pred_centers, dtype = float)
    nImg = len(gt_by_case)
    img_idxs = np.hstack([[i] * len(pred_by_case[i]) for i in range(nImg)]).astype(int)
    # pdb.set_trace()
    pred_bbox_nx7 = np.vstack(pred_by_case)

    orders = np.argsort(pred_bbox_nx7[:, -1])[::-1]
    pred_bbox_nx7 = pred_bbox_nx7[orders, :]
    img_idxs = img_idxs[orders]
    gt_bbox_nx6 = np.vstack(gt_by_case)
    remove_zeros = 0
    for l in range(len(gt_bbox_nx6)):
        if np.all(gt_bbox_nx6[l] == 0):
            remove_zeros += 1
    nGt = len(gt_bbox_nx6) - remove_zeros

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gt_by_case]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    # pdb.set_trace()
    for i in tqdm(range(len(pred_bbox_nx7))):
        # pdb.set_trace()
        inside = check_center_inside(pred_bbox_nx7[i][:6], gt_nx6=gt_by_case[img_idxs[i]][:, :6] )
        # print(f'Prob {i} ', centers_cat[i], gt_boxes[img_idxs[i]], inside)
        if 1 not in inside:
            nMiss += 1
        else:
            for j in range(len(inside)):
                if inside[j] == 1 and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)
        # pdb.set_trace()
    sens = np.array(tps, dtype=float) / nGt
    fps = np.array(fps, dtype=float) / nImg

    if luna_output_format:
        fps_itp = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 50, 1000], dtype=np.float)
        sens_itp = np.interp(fps_itp, fps, sens)
        sens_mean = sens_itp.mean()
    else:
        fps_itp = fps
        sens_itp = sens
        sens_mean = sens.mean()

    if plt_figure:
        fig = plt.figure(figsize=(16, 8))
        plt.plot(fps, sens, color='b', lw=2)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.125, 8])
        plt.ylim([0, 1.1])
        plt.xlabel('Average number of false positives per scan')  # 横坐标是fpr
        plt.ylabel('True Positive Rate')  # 纵坐标是tpr
        plt.title('FROC performence')
    else: fig = None

    result_dict = {}
    result_dict['0.125'] = sens_itp[0]
    result_dict['0.25'] = sens_itp[1]
    result_dict['0.5'] = sens_itp[2]
    result_dict['1.0'] = sens_itp[3]
    result_dict['2.0'] = sens_itp[4]
    result_dict['4.0'] = sens_itp[5]
    result_dict['8.0'] = sens_itp[6]
    result_dict['50.0'] = sens_itp[7]
    result_dict['1000.0'] = sens_itp[8]
    result_dict['mean'] = sens_mean
    return result_dict, fig


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
