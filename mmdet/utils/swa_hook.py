import torch
from mmcv.runner import HOOKS, Hook, ClosureHook
# inspiration https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/


@HOOKS.register_module()
class SWAOptimizer(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

loader, optimizer, model, loss_fn = ...
# fine-tuning pretrained model with SWA
# create a new hook to store and update swa_model.
# using CosineRestartLrUpdaterHook or CyclicLrUpdaterHook 
swa_model = AveragedModel(model) # apply during initialization 
scheduler = CosineAnnealingLR(optimizer, T_max=100) # change to lR update hook by mmcv 
swa_start = 5
swa_scheduler = SWALR(optimizer, swa_lr=0.05) 
# change to lR update hook by mmcv: CosineRestartLrUpdaterHook or 

for epoch in range(100):
      for input, target in loader:
          optimizer.zero_grad()
          loss_fn(model(input), target).backward()
          optimizer.step()

      if epoch > swa_start: # on epoch end
          swa_model.update_parameters(model)
          swa_scheduler.step()
      else:
          scheduler.step()

# Update bn statistics for the swa_model at the end
# torch.optim.swa_utils.update_bn(loader, swa_model)
# Use swa_model to make predictions on test data 
# preds = swa_model(test_input)