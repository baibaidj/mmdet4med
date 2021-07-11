import numpy as np
from numpy.lib.npyio import load
from .load_nn import load_pickle
from ..builder import PIPELINES


@PIPELINES.register_module()
class Load1CaseNN:

    def __init__(self, np_load_mode = 'r+') -> None:
        self.memmap_mode = np_load_mode
        pass

    def __call__(self, results):
        results['img'] = np.load(results['img_fp'], self.memmap_mode, allow_pickle=True)
        results['gt_instance_seg'] = np.load(results['label_fp'], self.memmap_mode, allow_pickle=True)
        results['property'] = load_pickle(results['property_fp'])
        results['bboxes'] = load_pickle(results['bboxes_fp'])
        return results