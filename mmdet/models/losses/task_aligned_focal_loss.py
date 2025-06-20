import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss, print_tensor
import ipdb 

# python version no_sigmoid
def focal_loss_with_prob(prob,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    target_one_hot = prob.new_zeros(len(prob), len(prob[0]) + 1).scatter_(1, target.unsqueeze(1), 1)[:, :-1]

    flatten_alpha = torch.empty_like(prob).fill_(1 - alpha)
    flatten_alpha[target_one_hot == 1] = alpha

    pt = torch.where(target_one_hot == 1, prob, 1 - prob)

    ce_loss = F.binary_cross_entropy(
        prob, target_one_hot, reduction='none')
    loss = flatten_alpha * torch.pow(1 - pt, gamma) * ce_loss
    # valid_mask = weight > 0
    # print('focal loss with prob')
    # valid_mask = weight != 0
    # valid_target = target[valid_mask]
    # valid_prob = prob[valid_mask]
    # valid_ce = ce_loss[valid_mask]
    # valid_pt = pt[valid_mask]
    # valid_alpha = flatten_alpha[valid_mask]
    # valid_loss = loss[valid_mask]
    # ipdb.set_trace()
    if weight is not None:
        loss = weight_reduce_loss(loss, weight.reshape(-1, 1), reduction, avg_factor)
    else:
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss

# python version for task aligned focal loss
def task_aigned_focal_loss(prob,
                       target,
                       alignment_metric,
                       weight=None,
                       gamma=2.0,
                       reduction='mean',
                       avg_factor=None):
    target_one_hot = prob.new_zeros(len(prob), len(prob[0]) + 1).scatter_(1, target.unsqueeze(1), 1)[:, :-1]
    soft_label = alignment_metric.unsqueeze(-1) * target_one_hot

    ce_loss = F.binary_cross_entropy(
        prob, soft_label, reduction='none')
    loss = torch.pow(torch.abs(soft_label - prob), gamma) * ce_loss

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss


@LOSSES.register_module()
class FocalLossWithProb(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss. Defaults to 0.25. 
                weighting foreground class and background class.if 0.5 then fg and bg have same weight. 
                if you want fg has larger weight, then alpha should be bigger than 0.5. 
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLossWithProb, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * focal_loss_with_prob(
                prob,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls



@LOSSES.register_module()
class TaskAlignedFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(TaskAlignedFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                prob,
                target,
                alignment_metric,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * task_aigned_focal_loss(
                prob,
                target,
                alignment_metric,
                weight,
                gamma=self.gamma,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
