
import torch
import torch.nn as nn
from ..builder import LOSSES
import copy, pdb
from .focal_loss import FocalLoss
from .cross_entropy_loss import *
from .seg_loss import *

@LOSSES.register_module()
class ComboLossMed(nn.Module):
    """ComboLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=(1.0, 0.5),
                 focal_loss_gamma = None, #dict(gamma = 2.0, alpha = 0.25, weight=None, reduction='mean'), 
                 num_classes = 1,
                 dice_cfg = dict(alpha = 1.0, beta = 1.0, ignore_0 = True, centerline_dice_weight = 0), 
                 group_dict =  None, 
                 group_loss_weight = 0.5,
                 verbose = False,
                 use_marginal = False,
                 ):
        super(ComboLossMed, self).__init__()

        assert len(loss_weight) == 2
        self.reduction = reduction
        self.loss_weights = loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.class_weight_origin = class_weight
        # self.uncertain_map_alpha = uncertain_map_alpha
        # self.dice_alpha = dice_alpha
        # self.dice_beta = dice_beta
        self.verbose = verbose
        self.group_dict = group_dict
        self.group_loss_weight = group_loss_weight
        # self.pos_topk = pos_topk
        self.num_classes = num_classes
        self.cls_out_channels = max(num_classes, 2)
        self.num_classes4loss = 2 if use_marginal else num_classes
        self.dice_cfg = dice_cfg
        self.use_marginal = use_marginal

        self.criterion_list = []

        for i, lw in enumerate(self.loss_weights):
            if lw ==0: criterion_i = None
            else:
                if i == 0:
                    if self.focal_loss_gamma:
                        criterion_i = FocalLoss(alpha = self.class_weight_origin, gamma= self.focal_loss_gamma)
                    else:
                        criterion_i = cross_entropy #if num_classes > 1 else binary_ce_general
                else:
                    criterion_i = SoftDiceLoss(self.cls_out_channels, **dice_cfg) 
            self.criterion_list.append(criterion_i)

    def forward_inner(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # the diceloss will do inplace operation on gt. In cascade scenario, GT is needed in multiple decoder heads to compute gradient
        # where inplace operation is not allowed. this issue can be solved by cloning GT every time the loss fucntion is called.
        label = label.clone().detach()

        if self.class_weight is not None: 
            self.class_weight = torch.tensor(self.class_weight, device=cls_score.device, dtype= cls_score.dtype)
            # if self.verbose: print(f'[Loss] class weight', self.class_weight)

        loss_1 = self.loss_weights[0] * self.criterion_list[0](
                                            cls_score,
                                            label,
                                            weight,
                                            class_weight=self.class_weight.clone().detach(),
                                            reduction=reduction,
                                            avg_factor=avg_factor,
                                            **kwargs) if self.loss_weights[0] != 0 else 0
        # pdb.set_trace()
        if reduction == 'none':
            loss_2 = 0.0
        else:
            loss_2 = self.loss_weights[1] * self.criterion_list[1](
                        cls_score, 
                        label, 
                        weight, 
                        class_weight = self.class_weight.clone().detach()) if self.loss_weights[1] != 0 else 0

        if self.verbose: print_tensor('[ComboLoss] BCE', loss_1)
        if self.verbose and isinstance(loss_2, torch.Tensor): print_tensor('[ComboLoss] Dloss', loss_2)
        total_loss = loss_1 + loss_2 
        return total_loss

    def forward(self, cls_score, label, *args, **kwargs):

        # pdb.set_trace(header=...)
        if self.verbose:
            with torch.no_grad():
                print_tensor('[ComboLoss] predscore', cls_score)
                # print_tensor('predmask', cls_score.argmax(1) if self.num_classes > 1 else cls)
                print_tensor('[ComboLoss] truemask', label)

        self.class_weight = copy.deepcopy(self.class_weight_origin)
        if self.loss_weights[1] !=0: self.criterion_list[1].update1hot_encoder(self.cls_out_channels)
        loss_main = self.forward_inner(cls_score, label, *args, **kwargs)

        if self.group_dict is not None:
            if self.verbose: print('Combine class and compute loss')
            cls_score = group_onehot_prob(cls_score, self.group_dict)
            label = group_class_decimal(label, self.group_dict)
            self.criterion_list[1].update1hot_encoder(self.cls_out_channels if self.use_marginal else len(self.group_dict))
            self.class_weight = [sum([self.class_weight_origin[i] for i in self.group_dict[nc]])/len(self.group_dict[nc])
                                 for nc in list(self.group_dict)]            
            if self.verbose: print(f'\tNew num class {self.criterion_list[1].num_classes}, class weight {self.class_weight}')
            loss_main += self.group_loss_weight* self.forward_inner(cls_score, label, *args, **kwargs)
        return loss_main

