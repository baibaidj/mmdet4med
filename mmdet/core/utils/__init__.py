# Copyright (c) OpenMMLab. All rights reserved.
from .dist_utils import (DistOptimizerHook, all_reduce_dict, allreduce_grads,
                         reduce_mean)
from .misc import (center_of_mass, flip_tensor, generate_coordinate,
                   mask2ndarray, multi_apply, unmap)
from .swa_hook import SWAHook

print_tensor = lambda n, x: print(n, type(x), x.dtype, x.shape, x.min(), x.max())
__all__ = [
    'allreduce_grads', 'DistOptimizerHook', 'reduce_mean', 'multi_apply',
    'unmap', 'mask2ndarray', 'flip_tensor', 'all_reduce_dict',
    'center_of_mass', 'generate_coordinate', 
    'SWAHook'
]
