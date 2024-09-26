import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CELoss, FocalLoss, SwinCombinedLoss

class LossSelector:
    def __init__(self, loss_name='CE', **kwargs):
        self.loss_name = loss_name
        self.kwargs = kwargs
        self.loss_fn = self.get_loss(self.loss_name, **self.kwargs)

    def __call__(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def get_loss(self, loss_name, **kwargs):
        if loss_name == 'CE':
            return CELoss(kwargs.get('label_smoothing', 0.0))
        elif loss_name == 'Focal':
            return FocalLoss(kwargs.get('alpha_val', 0.25), kwargs.get('gamma_val', 2.0))
        elif loss_name == 'swin_loss':
            return SwinCombinedLoss()
        else:
            raise ValueError(f"'{loss_name}' is not a valid loss name")