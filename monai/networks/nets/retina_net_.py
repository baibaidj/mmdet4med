"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

from monai.networks.blocks.nms_ import nms_2D, nms_3D, apply_box_deltas_2D, apply_box_deltas_3D, unique1d, clip_to_window
from monai.networks.blocks.anchors_ import generate_pyramid_anchors
from monai.networks.layers.fpn_ import FPN
from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Norm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

############################################################
#  Network Heads
############################################################

def refine_detections(
        dim, anchors, probs, deltas, batch_ixs,
        patch_size,
        rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
        detection_nms_threshold = 1e-5,  # needs to be > 0, otherwise all predictions are one cluster.,
        model_max_instances_per_batch_element=30  # per batch element and class.
    ):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    # pre-selection of detections for NMS-speedup. per entire batch.
    pre_nms_limit = 10000 if dim == 2 else 50000
    window = np.array([0, 0, patch_size[0], patch_size[1], 0, patch_size[2]])
    scale = np.array([patch_size[0], patch_size[1], patch_size[0], patch_size[1], patch_size[2], patch_size[2]])
    if torch.cuda.is_available():
        anchors = anchors.repeat(len(torch.unique(batch_ixs)), 1)
    else:
        anchors = anchors.repeat(len(np.unique(batch_ixs)), 1)

    if dim == 2:
        rpn_bbox_std_dev = rpn_bbox_std_dev[:4]
        # bbox_std_dev = bbox_std_dev[:4]
        window = window[:4]
        scale = scale[:4]

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)
    keep_arr = keep_arr.long()

    pre_nms_scores = flat_probs[:pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long()

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(rpn_bbox_std_dev, [1, dim * 2])).float()
    scale = torch.from_numpy(scale).float()

    if torch.cuda.is_available():
        keep = keep.cuda()
        std_dev = std_dev.cuda()
        scale = scale.cuda()

    if dim == 2:
        refined_rois = apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale
    else:
        refined_rois = apply_box_deltas_3D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = clip_to_window(window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores

            if dim == 2:
                class_keep = nms_2D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), detection_nms_threshold)
            else:
                class_keep = nms_3D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), detection_nms_threshold)

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:model_max_instances_per_batch_element]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result


def refine_det(class_logits, bb_outputs,
               dim, anchors, patch_size, rpn_bbox_std_dev,
               detection_nms_threshold, model_max_instances_per_batch_element):
    # merge batch_dimension and store info in batch_ixs for re-allocation.
    batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1)
    if torch.cuda.is_available():
        batch_ixs = batch_ixs.cuda()
    flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
    flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
    detections = refine_detections(dim, anchors, flat_class_softmax, flat_bb_outputs, batch_ixs,
                                   patch_size,
                                   rpn_bbox_std_dev=rpn_bbox_std_dev,
                                   detection_nms_threshold=detection_nms_threshold,
                                   model_max_instances_per_batch_element=model_max_instances_per_batch_element)
    return detections


class Classifier(nn.Module):

    def __init__(
            self,
            dim,
            head_classes=3,
            in_channels=36,
            out_channels_per_pos=9,
            features=64,
            stride=1,
            act=Act.PRELU):
        """
        Builds the classifier sub-network.
        """
        super(Classifier, self).__init__()
        self.dim = dim
        self.n_classes = head_classes
        output_channels = out_channels_per_pos * head_classes

        self.conv_1 = Convolution(self.dim, in_channels, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_2 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_3 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_4 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_final = Convolution(self.dim, features, output_channels, kernel_size=3, strides=stride, padding=1, act=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)

        return [class_logits]


class BBRegressor(nn.Module):

    def __init__(
            self,
            dim,
            in_channels=36,
            out_channels_per_pos=9,
            features=64,
            stride=1,
            act=Act.PRELU):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        self.dim = dim
        output_channels = out_channels_per_pos * self.dim * 2

        self.conv_1 = Convolution(self.dim, in_channels, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_2 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_3 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_4 = Convolution(self.dim, features, features, kernel_size=3, strides=stride, padding=1, act=act)
        self.conv_final = Convolution(self.dim, features, output_channels, kernel_size=3, strides=stride, padding=1, act=None)

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, self.dim * 2)

        return [bb_logits]


class RetinaNet(nn.Module):

    def __init__(self,
                 dim,
                 patch_size,
                 in_channels,
                 headers=2,
                 start_filts=18,
                 end_filts=36,
                 pyramid_levels=[0, 1, 2, 3],
                 act=Act.PRELU,
                 norm=Norm.INSTANCE,
                 res_architecture='resnet50',   # original_param = 'resnet101'
                 anchor_stride=1,
                 rpn_bbox_std_dev=np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]),
                 detection_nms_threshold=0.01,
                 model_max_instances_per_batch_element=2000,
                 ):
        super(RetinaNet, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.start_filts = start_filts
        self.end_filts = end_filts
        self.act = act
        self.norm = norm
        self.headers = headers
        self.res_architecture = res_architecture
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.detection_nms_threshold = detection_nms_threshold
        self.model_max_instances_per_batch_element = model_max_instances_per_batch_element
        self.np_anchors = generate_pyramid_anchors(self.dim, self.patch_size, pyramid_levels=self.pyramid_levels, anchor_stride=anchor_stride)
        self.anchors = torch.from_numpy(self.np_anchors).float()
        if torch.cuda.is_available():
            self.anchors = self.anchors.cuda()

        self.Fpn = FPN(
            dim=self.dim,
            n_channels=self.in_channels,
            start_filts=self.start_filts,
            end_filts=self.end_filts,
            n_latent_dims=0,
            act=self.act,
            norm=self.norm,
            res_architecture=self.res_architecture,
            sixth_pooling=False,
            operate_stride1=False,
        )

        self.Classifier = Classifier(
            dim=self.dim,
            head_classes=self.headers,
            in_channels=self.end_filts,
            out_channels_per_pos=9,
            features=64,
            stride=anchor_stride,
            act=act,
        )

        self.BBRegressor = BBRegressor(
            dim=self.dim,
            in_channels=self.end_filts,
            out_channels_per_pos=9,
            features=64,
            stride=anchor_stride,
            act=act,
        )


    def return_anchors(self):
        return self.anchors


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        :return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # Feature extraction
        fpn_outs = self.Fpn(img)
        seg_logits = None
        selected_fmaps = [fpn_outs[i] for i in self.pyramid_levels]

        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        for p in selected_fmaps:
            class_layer_outputs.append(self.Classifier(p))
            bb_reg_layer_outputs.append(self.BBRegressor(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]
        return class_logits, bb_outputs, seg_logits
        #
        # # merge batch_dimension and store info in batch_ixs for re-allocation.
        # batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1)
        # if torch.cuda.is_available():
        #     batch_ixs = batch_ixs.cuda()
        # flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        # flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])
        #
        # detections = refine_detections(self.dim, self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs,
        #                                self.patch_size,
        #                                rpn_bbox_std_dev=self.rpn_bbox_std_dev,
        #                                detection_nms_threshold=self.detection_nms_threshold,
        #                                model_max_instances_per_batch_element=self.model_max_instances_per_batch_element)
        # return detections, class_logits, bb_outputs, seg_logits


if __name__ == '__main__':
    tensor = torch.rand([1, 1, 96, 96, 96], dtype=torch.float).cuda()
    net = RetinaNet(dim=3, headers=3, patch_size=[96, 96, 96], in_channels=1)
    net = torch.nn.DataParallel(net).cuda()
    out = net(tensor)
    print(out)