import numpy as np
import matplotlib.pylab as plt


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
        if len(gt_bbox) == 0:
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

        if x >= gtx1 and x <= gtx2 and y >= gty1 and y <= gty2 and z >= gtz1 and z <= gtz2:
            inside_tag.append(1)
        else:
            inside_tag.append(0)
    return inside_tag

def check_center_inside(pred_center, gt_bboxes):
    x = pred_center[0]
    y = pred_center[1]
    z = pred_center[2]
    inside_tag = []
    for n in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[n]
        if len(gt_bbox) == 0:
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

        if x >= gtx1 and x <= gtx2 and y >= gty1 and y <= gty2 and z >= gtz1 and z <= gtz2:
            inside_tag.append(1)
        else:
            inside_tag.append(0)
    return inside_tag

def calculate_FROC(gt_boxes, pred_boxes, luna_output_format=True, plt_figure=False):
    nImg = len(gt_boxes)
    img_idxs = np.hstack([[i] * len(pred_boxes[i]) for i in range(nImg)]).astype(int)
    pred_boxes = np.vstack(pred_boxes)
    orders = np.argsort(pred_boxes[:, 6])[::-1]
    boxes_cat = pred_boxes[orders, :6]
    img_idxs = img_idxs[orders]

    gt_vstack = np.vstack(gt_boxes)
    remove_zeros = 0
    for l in range(len(gt_vstack)):
        if np.all(gt_vstack[l] == 0):
            remove_zeros += 1
    nGt = len(gt_vstack) - remove_zeros
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gt_boxes]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        inside = check_bbox_inside(boxes_cat[i], gt_bboxes=gt_boxes[img_idxs[i]])
        if 1 not in inside:
            nMiss += 1
        else:
            for j in range(len(inside)):
                if inside[j] == 1 and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)

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
        plt.plot(fps, sens, color='b', lw=2)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.125, 8])
        plt.ylim([0, 1.1])
        plt.xlabel('Average number of false positives per scan')  # 横坐标是fpr
        plt.ylabel('True Positive Rate')  # 纵坐标是tpr
        plt.title('FROC performence')

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
    return result_dict

def calculate_FROC_by_center(gt_boxes, pred_centers, luna_output_format=True, plt_figure=False):
    nImg = len(gt_boxes)
    img_idxs = np.hstack([[i] * len(pred_centers[i]) for i in range(nImg)]).astype(int)
    pred_centers = np.vstack(pred_centers)
    orders = np.argsort(pred_centers[:, 3])[::-1]
    centers_cat = pred_centers[orders, :3]
    img_idxs = img_idxs[orders]

    gt_vstack = np.vstack(gt_boxes)
    remove_zeros = 0
    for l in range(len(gt_vstack)):
        if np.all(gt_vstack[l] == 0):
            remove_zeros += 1
    nGt = len(gt_vstack) - remove_zeros
    hits = [np.zeros((len(gts),), dtype=bool) for gts in gt_boxes]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(centers_cat)):
        inside = check_center_inside(centers_cat[i], gt_bboxes=gt_boxes[img_idxs[i]])
        if 1 not in inside:
            nMiss += 1
        else:
            for j in range(len(inside)):
                if inside[j] == 1 and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)

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
        plt.plot(fps, sens, color='b', lw=2)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.125, 8])
        plt.ylim([0, 1.1])
        plt.xlabel('Average number of false positives per scan')  # 横坐标是fpr
        plt.ylabel('True Positive Rate')  # 纵坐标是tpr
        plt.title('FROC performence')

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
    return result_dict