
import torch
import torch.nn.functional as F

from mmdet.utils.resize import resize_3d
from ..builder import DETECTORS, build_backbone, build_neck, build_head
from .base_learner_3d import BaseLearner3D
from ...datasets.pipelines import Compose
from mmdet.models.necks.fpn_3d import FPN3D
from mmdet.models.semantic_heads import FCNHead3D
from mmdet.models.backbones.swin_3d import SwinTransformer3D4SimMIM
from mmdet.utils import print_tensor
from torch import distributed as dist
import ipdb

@DETECTORS.register_module()
class SimMIM(BaseLearner3D):
    def __init__(self, 
                 backbone,
                 neck=None,
                 seg_head = None,
                 gpu_aug_pipelines = [],
                 train_cfg=None,
                 test_cfg=None,
                 mask_cfg = dict(input_size=(160, 160, 160), mask_patch_size=32,
                                 stem_stride=4, mask_ratio=0.6), 
                init_cfg = None, 
                verbose = False):
        super(SimMIM).__init__(init_cfg)

        self.backbone : SwinTransformer3D4SimMIM = build_backbone(backbone)
        if neck is not None:
            self.neck : FPN3D = build_neck(neck)

        self.seg_head: FCNHead3D = build_head(seg_head) if seg_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.verbose = verbose
        self.in_chans = self.backbone.in_channels
        self.stem_stride = self.backbone.strides[0]
        self.mask_generator = MaskGeneratorND(**mask_cfg)

        self.gpu_pipelines = Compose(gpu_aug_pipelines)


    @torch.no_grad()
    def update_img_metas(self, imgs, img_metas, **kwargs):
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        mini_batch_holder = {'img': imgs}
        mini_batch_holder[f'img_meta_dict'] = [a[f'img_meta_dict'] for a in img_metas]
        data_dict = self.gpu_pipelines(mini_batch_holder)
        imgs = data_dict['img']
        device = imgs.device
        mask_list = []
        for i, img_meta in enumerate(img_metas):
            mask_i = self.mask_generator(device)
            mask_list.append(mask_i)
        masks = torch.stack(mask_list)
        return imgs, masks


    def forward_train(self, img, img_metas, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        rank = dist.get_rank()
        # im_q = img[:, 0, ...].contiguous() # BCHWD by one transform
        # im_k = img[:, 1, ...].contiguous() # BCHWD by another transform
        x, mask = self.update_img_metas(img, img_metas)

        if self.verbose and rank == 0:
            print_tensor('\nIMG raw', x)
            print_tensor('IMG mask', mask)

    # def forward(self, x, mask : torch.Tensor):
        feat = self.extract_feat(x, mask)
        x_rec = self.seg_head.forward_test(feat, img_metas, self.test_cfg)
        if x_rec.shape != x.shape: 
            x_rec = resize_3d(x_rec, size = x.shape[2:], mode = 'trilinear')
        # ops = torch.repeat_interleave()


        mask = mask.repeat_interleave(self.stem_stride, 1).repeat_interleave(
                                    self.stem_stride, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        losses = {'loss_rec' : loss}
        return losses


    def extract_feat(self, img, mask = None):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img, mask = mask)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def simple_test(self, img, img_metas, mask = None):
        """Test without augmentation."""
        z = self.extract_feat(img, mask)
        x_rec = self.seg_head.forward_test(z, img_metas, self.test_cfg)
        x_rec = resize_3d(x_rec, size = img.shape[2:], mode = 'trilinear')

        return x_rec


    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def product(vector):
    result = 0
    if isinstance(vector, (int, float)):
        return vector
    elif isinstance(vector, (list, tuple)):
        num_ = len(vector)
        if num_ == 0:
            return
        elif num_ == 1:
            return vector[1]
        else:
            result = vector[0]
            for n in vector[1:]:
                result = result * n
    else:
        raise ValueError(f'Input should be number but got {type(vector)}')
    
import numpy as np

class MaskGeneratorND:
    def __init__(self, input_size=(192, 192), 
        mask_patch_size=32, stem_stride=4, mask_ratio=0.6, device = 'cpu'):
        """
        ND: N-dimensional, N can assume 2 or 3
        Args:
            model_patch_size: the stride of stem layer in Swin
        """
        assert isinstance(input_size, (tuple, list)), \
            f'input size should be tuple or list but got {type(input_size)}'

        # self.add_key = add_key
        # self.img_key = img_key
        self.input_size = input_size
        self.dim = len(input_size)
        self.num_pixel = product(input_size)
        self.mask_patch_size = mask_patch_size
        self.stem_stride = stem_stride
        self.mask_ratio = mask_ratio
        self.device = device
        
        assert self.num_pixel % self.mask_patch_size == 0
        assert self.mask_patch_size % self.stem_stride == 0
        
        self.token_map_size = tuple([s // self.mask_patch_size for i, s in enumerate(self.input_size)])
        self.scale = self.mask_patch_size // self.stem_stride
        
        self.token_count = product(self.token_map_size)
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))


    def __call__(self, device = 'cuda:0'):
        device = device if device is not None else self.device
        mask = torch.zeros(self.token_count, dtype=torch.long, device = device)
        mask_idx = torch.randperm(self.token_count, device=device)[:self.mask_count]
        mask[mask_idx] = 1
        
        mask = mask.reshape(self.token_map_size) # 6x6?
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        if self.dim == 3: 
            mask = mask.repeat(self.scale, axis = 2) # 48x48?
        
        return mask
