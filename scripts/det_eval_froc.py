import os
import sys
import numpy as np
import pandas as pd
from scipy import interpolate

import torch



def check_bbox_inside(pred_bbox, gt_bboxes):
    x = (pred_bbox[0] + pred_bbox[3]) / 2
    y = (pred_bbox[1] + pred_bbox[4]) / 2
    z = (pred_bbox[2] + pred_bbox[5]) / 2

    inside_tags = []
    for n in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[n]
        gtx1 = gt_bbox[0]
        gty1 = gt_bbox[1]
        gtz1 = gt_bbox[2]
        gtx2 = gt_bbox[3]
        gty2 = gt_bbox[4]
        gtz2 = gt_bbox[5]

        if x >= gtx1 and x <= gtx2 and y >= gty1 and y <= gty2 and z >= gtz1 and z <= gtz2:
        #if x > gtx1 - 3 and x < gtx2 + 3 and y > gty1 - 3 and y < gty2 + 3 and z > gtz1 - 3 and z < gtz2 + 3:

            inside_tags.append(1)
        else:
            inside_tags.append(0)

    return inside_tags


def calculate_froc():
    data_path = cfg.preprocess_path
    train_data, val_data, train_data_dict, val_data_dict, train_nodule_num, val_nodule_num = get_split_dataset(data_path, cfg.tasks, cfg.train_fold, cfg.val_fold)
    print("train data: ", len(train_data), "train nodule: ", train_nodule_num, "val data: ", len(val_data), "val nodule: ", val_nodule_num)

    csv_path = os.path.join(cfg.output_path, cfg.output_filename)
    print("evaluate file: ", csv_path)
    df_pred = pd.read_csv(csv_path)


    pred_boxes = []
    gt_boxes = []
    for (task, filename), group in df_pred.groupby(['task', 'filename']):
        key = task + '_' + filename
        if task not in cfg.tasks or key not in val_data_dict.keys():
            continue
        bbox = val_data_dict[key]
        gt_boxes.append(bbox)

        pred_list = []
        for index, row in group.iterrows():
            x1, y1, z1, x2, y2, z2, prob = row['x1'], row['y1'], row['z1'], row['x2'], row['y2'], row['z2'], row['detect_prob']
            pred_list.append(np.array([x1, y1, z1, x2, y2, z2, prob], dtype=np.float32))
        pred = np.asarray(pred_list)
        pred_boxes.append(pred)


    nImg = len(gt_boxes)
    img_idxs = np.hstack([[i] * len(pred_boxes[i]) for i in range(nImg)]).astype(int)
    pred_boxes = np.vstack(pred_boxes)
    orders = np.argsort(pred_boxes[:, 6])[::-1]
    boxes_cat = pred_boxes[orders, :6]
    boxes_prob = pred_boxes[orders, 6]
    img_idxs = img_idxs[orders]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gt_boxes]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    probs = []
    for i in range(len(boxes_cat)):
        inside_tags = check_bbox_inside(boxes_cat[i], gt_bboxes=gt_boxes[img_idxs[i]])
        if sum(inside_tags) == 0 or (gt_boxes[img_idxs[i]].shape[0] == 1 and np.sum(gt_boxes[img_idxs[i]]) == 0):
            nMiss += 1
        else:
            for j in range(len(inside_tags)):
                if inside_tags[j] == 1 and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1
        tps.append(nHits)
        fps.append(nMiss)
        probs.append(boxes_prob[i])

    nGt = sum([gts.shape[0] for gts in gt_boxes if not (gts.shape[0] == 1 and np.sum(gts) == 0)])
    sens = np.array(tps, dtype=float) / nGt
    fps = np.array(fps, dtype=float) / nImg

    fps_itp = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8, 50, 100, 200, 500, 1000], dtype=np.float)

    sens_itp = np.interp(fps_itp, fps, sens)
    sens_mean = sens_itp[0:7].mean()

    result_dict = {}
    result_dict['0.125'] = sens_itp[0]
    result_dict['0.25'] = sens_itp[1]
    result_dict['0.5'] = sens_itp[2]
    result_dict['1.0'] = sens_itp[3]
    result_dict['2.0'] = sens_itp[4]
    result_dict['4.0'] = sens_itp[5]
    result_dict['8.0'] = sens_itp[6]
    result_dict['50.0'] = sens_itp[7]
    result_dict['100.0'] = sens_itp[8]
    result_dict['200.0'] = sens_itp[9]
    result_dict['500.0'] = sens_itp[10]
    result_dict['1000.0'] = sens_itp[11]
    result_dict['mean'] = sens_mean

    prob_itp = np.interp(sens_itp, sens, probs)
    return result_dict, prob_itp


def main():
    froc_dict, prob_itp = calculate_froc()
    froc = ["%.3f" % value for value in froc_dict.values()]
    froc_prob = ["%.3f" % value for value in prob_itp]

    print("froc: ", froc)
    print("prob: ", froc_prob)



if __name__ == "__main__":
    main()