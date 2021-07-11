"""
Copyright 2020 KEYA Medical Algorithm Team
Version: 1.0
"""

import os
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def load_model(net, load_dir, gpu_ids=None):
    def partial_weight_update(model, pretrained_state):
        model_state = model.state_dict()
        pretrained_state = align_model_parameter(pretrained_state, model_state)

        state_dict = {k: v for k, v in pretrained_state.items()
                      if k in model_state.keys()}
        model_state.update(state_dict)
        model.load_state_dict(model_state)
        return model

    checkpoint = load_dir
    if isinstance(load_dir, str):
        checkpoint = torch.load(load_dir)
    try:
        partial_weight_update(net, checkpoint['state_dict'])
    except Exception as e:
        print(e)
    # net.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint.get('epoch', 0)
    lr = checkpoint.get('lr', 1e-3)
    best_metric = checkpoint.get('best_metric', -1)
    # # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # if gpu_ids is None:  # use all gpus in default
    #     net = torch.nn.DataParallel(net) #.cuda()
    # else:
    #     net = torch.nn.dataParallel(
    #         net, device_ids=list(range(len(gpu_ids)))).cuda()
    # # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    # #     net = net.cuda()
    return net, int(epoch), float(lr), float(best_metric)


def save_model(net, epoch, save_dir, tag=None, save_best=False, lr=1e-3, best_metric=-1):
    folder = os.path.exists(save_dir)
    if not folder:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if save_best:
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'lr': lr,
            'best_metric': best_metric,
            'state_dict': state_dict},
            os.path.join(save_dir, f"{tag}_model_best.pth"))
    else:  # save last
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'lr': lr,
            'best_metric': best_metric,
            'state_dict': state_dict},
            os.path.join(save_dir, f'{tag}_model_last.pth'))


def build_network(net, checkpoint_path=None, snapshot=None, gpu_ids=None, just_weight = False, put_gpu = True):

    epoch = 0
    lr = 1e-3
    best_metric = -1

    if snapshot is not None and checkpoint_path is not None:
        ck_fp = os.path.join(checkpoint_path, snapshot)
        print('Load checkpoint from ', ck_fp)
        net, epoch, lr, best_metric = load_model(net, ck_fp, gpu_ids)
        if just_weight: epoch = 0; lr = 1e-3

    if put_gpu:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if gpu_ids is None:  # use all gpus in default
                net = torch.nn.DataParallel(net).cuda()
            else:
                net = torch.nn.DataParallel(net, device_ids=gpu_ids, output_device=gpu_ids).cuda()
                # if len(gpu_ids) > 1:
                #     # initialize the distributed training process, every GPU runs in a process
                #     dist.init_process_group(backend="nccl", init_method="env://")
                #     # net = torch.nn.
                # else: 
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
            net = net.cuda()
    return net, epoch, lr, best_metric


def load_pretrain_model(net, checkpoint, gpu_ids=None):
    def partial_weight_update(model, pretrained_state):
        model_state = model.state_dict()
        pretrained_state = align_model_parameter(pretrained_state, model_state)

        state_dict = {
            k: v
            for k, v in pretrained_state.items() if k in model_state.keys()
        }
        model_state.update(state_dict)
        try:
            model.load_state_dict(model_state)
        except Exception as e:
            print(e)
        return model

    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint)
    partial_weight_update(net, checkpoint['state_dict'])
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def align_model_parameter(src_state, dst_state):
    """
    align parameter key in src network to dst network.
    DP and DDP model's parameter key start with "module." but other's not!
    """
    dst_key = list(dst_state.keys())[0]
    src_key = list(src_state.keys())[0]
    if dst_key.startswith('module.') and not src_key.startswith('module.'):
        src_state = {f'module.{k}': v for k, v in src_state.items()}
    elif not dst_key.startswith('module.') and src_key.startswith('module.'):
        src_state = {k[7:]: v for k, v in src_state.items()}
    return src_state
