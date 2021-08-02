import numpy as np
import torch


print_tensor = lambda n, x: print(n, type(x), x.shape, x.min(), x.max())
dice_func = lambda i, u, eps: (2 * i + eps * (u == 0)) / (i + u + eps)

def cfsmat4mask_batched(results, gt_seg_maps, num_classes, ignore_index = 255, eps = 1e-5):
    """Calculate confusin matrix as basis for further metrics

    Args:
        results (list[ndarray]): List of prediction segmentation maps as int 
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs

    cfs_mat_list = []

    for i in range(num_imgs):
        cfs_mat = semantic_confusion_matrix(results[i], gt_seg_maps[i], 
                                            num_classes, ignore_index)
        cfs_mat_list.append(cfs_mat)
    
    return cfs_mat_list


def semantic_confusion_matrix(label_pred, label_true, n_class, ignore_index, verbose = False):
    """

    supposs gt and pred representing by the actual class index rather than one-hot vector. 
    N representing number of class (e.g. 3). We want to make a nxn confusion matrix which stores 
    how many number of samples (points in the image case) are there under each pred/gt value combinations. 
    

    row: label_true, 
    col: label_pred; 
    n_class >= 2
    cfs_matrix = np.bincount(n_class * label_true + label_pred, 
                             minlength=n_class**2).reshape(n_class, n_class)
    for class i, fn = sum(cfs_matrix[i]) - cfs_matrix[i, i], fp = sum(cfs_matrix[:, i]) - cfs_matrix[i,i]

    example: 
    label_true = array([[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 2, 0],
                        [0, 0, 1, 1]])

    label_pred = array([[0, 0, 0, 0],                                                                                                                                          
                        [0, 0, 0, 0],                                                                                                                                          
                        [0, 0, 2, 1],                                                                                                                                          
                        [0, 0, 1, 1]])

    n_class = 3

    step by step computation results: 
    
        first_term = label_true.flatten() * n_class = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 3])
        second_term = label_pred.flatten() = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1])

        fuse_term = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 4, 4])

        bincount_result = array([12, 1, 0, 0, 2, 0, 0, 0, 1])

        cfs_matrix = array([[12,  1,  0],
                            [ 0,  2,  0],
                            [ 0,  0,  1]])

        if gt==pred, then gt*n + pred = diagnoal index of the nxn confusion matrix                    
        gt/pr  0  1   2
            0| 0   1   2
            1| 3   4   5
            2| 6   7   8

    """

    if isinstance(label_true, torch.Tensor):
        label_true = label_true.cpu().numpy()
    if isinstance(label_pred, torch.Tensor):
        label_pred = label_pred.cpu().numpy()

    n_class = max(n_class, 2)
    mask = (label_true >= 0) & (label_true < n_class) & (label_true != ignore_index)
    
    label_pred = label_pred[mask].astype(np.uint8)
    label_true = label_true[mask].astype(np.uint8)

    if verbose: print_tensor('\tpred', label_pred); print_tensor('\tgt',label_true)
    cfs_matrix = np.bincount(
        n_class * label_true.astype(int) + label_pred, 
        minlength=n_class**2
        ).reshape(n_class, n_class)
    return cfs_matrix
    

class Metric4ConfusionMatrix(object):
    """
        gt/pr  0  1   2
            0| 0   1   2
            1| 3   4   5
            2| 6   7   8
    """
    def __init__(self, cfs_matrix, class_i = None, eps = 1e-6) -> None:
        self.cfs_matrix = cfs_matrix
        self.eps = eps
        self.class_i = class_i

    def renew_class(self, class_i): 
        assert type(class_i) is int
        assert class_i < self.num_classes
        self.class_i = class_i

    @property
    def true_area(self): return sum(self.cfs_matrix[self.class_i, :])

    @property
    def pred_area(self): return sum(self.cfs_matrix[:, self.class_i])
     
    @property
    def intersection(self): return self.cfs_matrix[self.class_i, self.class_i]
    
    @property
    def union(self): return self.true_area + self.pred_area - self.intersection
    
    @property
    def tp(self): return self.intersection

    @property
    def tn(self): return self.total- self.true_area - self.pred_area + self.intersection

    @property
    def fn(self): return self.true_area - self.tp

    @property
    def fp(self): return self.pred_area - self.tp

    @property
    def iou(self): return (self.intersection + self.eps*(self.union == 0)) / (self.union + self.eps)

    @property
    def dice(self): 
        return dice_func(self.intersection, self.union, self.eps)
    
    @property
    def total(self): return self.cfs_matrix.sum()

    @property
    def fn_rate(self): return self.fn/self.total

    @property
    def fp_rate(self): return self.fp/self.total

    @property
    def num_classes(self): return self.cfs_matrix.shape[0]

    @property
    def all_acc(self): 
        diagnal_ix = list(range(self.num_classes))
        return self.cfs_matrix[diagnal_ix, diagnal_ix].sum()/self.total

    @property
    def acc(self): return (self.tp + self.tn)/  self.total

    @property
    def recall(self): return (self.tp + self.eps) / (self.true_area + self.eps)

    @property
    def precision(self): return (self.tp + self.eps)/ (self.pred_area + self.eps)

def compute_metric1(cm):
    """


    return:
        suppose that there are 6 classes
        {
            'iou': [0.9974 0.9686 0.9449 0.9488 0.9514 0.953],
            'dice': [0.9986 0.9686 0.9765 0.9673 0.9361 0.895]
            ...
        }
    """
    metric_names = ['iou', 'dice', 'acc', 'all_acc', 'recall', 'precision']
    metric_dict = {s:[] for s in metric_names}
    metric_obj = Metric4ConfusionMatrix(cm)
    num_classes = metric_obj.num_classes

    for i in range(num_classes):
        metric_obj.renew_class(i)
        for s in metric_names:
            # print('\tgetattr', s, getattr(metric_obj, s))
            metric_dict[s].append(getattr(metric_obj, s))
    return metric_dict

def metric_in_cfsmat_1by1(cfs_matrix_list):
    """
    give a list of confusion matrix for each slice
    construct a metric computer individually
    which can output a number of metrics for each class
    these metrics include 'iou', 'dice', 'acc', 'all_acc', 'recall', 'precision'
    
    output:
        metric3d: dict {iou: [class0, class1, ...],
                        dice: [class0, class1, ...]}

    """
    metric2d_list = [compute_metric1(cm) for cm in cfs_matrix_list ]
    cm_3d = np.stack(cfs_matrix_list, axis = 0).sum(0)
    metric3d = compute_metric1(cm_3d)

    return metric2d_list, metric3d