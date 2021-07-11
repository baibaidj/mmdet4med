import threading
from os import makedirs
from os.path import dirname, join, basename
import SimpleITK as sitk
import numpy as np
import monai
import os
import json
import torch
from torch import nn
import torch.distributed as dist
from monai.data.dataloader_ import DataLoaderGPU


class PolyLR:
    def __init__(self, optimizer, initial_lr, max_epochs, lr_reduce):
        self.optimizer = optimizer
        if initial_lr is None:
            initial_lr = optimizer.param_groups[0]['lr']
        self.initial_lr = initial_lr
        self.max_epochs = max_epochs
        self.lr_reduce = lr_reduce
        self.epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        lr = self.initial_lr * (1 - epoch / self.max_epochs)**self.lr_reduce
        self.optimizer.param_groups[0]['lr'] = lr
        self.epoch += 1


def thread_save_tensor(data, fname):
    makedirs(dirname(fname), exist_ok=True)
    threading.Thread(target=sitk.WriteImage,
                     args=(sitk.GetImageFromArray(
                         data.to('cpu').numpy().squeeze()), fname)).start()


def step_save(data, folder, mode, metric=None):
    out = 'train_output'
    if mode == 'train':
        return

    out = 'val_output'
    for case, image, truth, predict in zip(data['case'], data['image'],
                                           data['label'], data['predict']):
        thread_save_tensor(image, join(folder, out, f'{case}_image.nii.gz'))
        thread_save_tensor(truth, join(folder, out, f'{case}_truth.nii.gz'))
        thread_save_tensor(predict, join(folder, out,
                                         f'{case}_predict.nii.gz'))


class DataLoader:
    def __init__(self,
                 params,
                 train_files,
                 val_files,
                 test_files=None,
                 train_weights=None):
        self.params = params
        self.train_files = train_files
        self.val_files = val_files
        self.train_weights = train_weights
        self.test_files = test_files

    def __call__(self, augmentation, mode):
        wsize = 1
        rank = 0
        if dist.is_initialized():
            wsize = dist.get_world_size()
            rank = dist.get_rank()

        # define dataset
        sampler = None
        if mode == 'train':
            shuffle = True
            files = self.train_files
            batch_size = self.params['batch_size']
            if self.train_weights is not None:
                from torch.utils.data import WeightedRandomSampler
                sampler = WeightedRandomSampler(self.train_weights,
                                                len(self.train_weights))
                shuffle = False

        elif mode == 'val':
            shuffle = False
            files = self.val_files[rank::wsize]
            batch_size = 1
        elif mode == 'val_all':
            shuffle = False
            files = (self.train_files + self.val_files)[rank::wsize]
            batch_size = 1
        elif mode == 'test':
            shuffle = False
            files = self.test_files[rank::wsize]
            batch_size = 1
        else:
            raise ValueError(f'mode{mode} is not supported')

        if isinstance(augmentation, dict):
            aug_cpu = augmentation.get('cpu')
        else:
            aug_cpu = augmentation
        type = self.params['cache_type']
        if type == 'normal':
            ds = monai.data.Dataset(data=files, transform=aug_cpu)
        elif type == 'cache':
            ds = monai.data.CacheDataset(data=files, transform=aug_cpu)
        elif type == 'smart_cache':
            ds = monai.data.SmartCacheDataset(data=files, transform=aug_cpu)
        else:
            raise ValueError(f'wrong type {type}')

        pin_memory = False if torch.cuda.is_available(
        ) is False else self.params['pin_memory']

        aug_gpu = None
        try:
            aug_gpu = augmentation.get('gpu')
        except:
            pass
        if aug_gpu is None:
            data_loader = monai.data.DataLoader(
                dataset=ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.params['num_workers'],
                pin_memory=pin_memory,
                sampler=sampler)
        else:
            data_loader = DataLoaderGPU(aug_gpu=aug_gpu,
                                        dataset=ds,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=self.params['num_workers'],
                                        pin_memory=pin_memory,
                                        sampler=sampler)
        return ds, data_loader
