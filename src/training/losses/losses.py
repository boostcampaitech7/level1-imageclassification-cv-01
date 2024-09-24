import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CELoss
from . import FocalLoss
from . import SwinCombinedLoss

class LossFactory:
    def __init__(self):
        # 초기화 단계에서는 특별한 동작 없음
        pass

    def __call__(self, loss_name='CE', **kwargs):
        return self.get_loss(loss_name, **kwargs)

    def get_loss(self, loss_name, **kwargs):
        if loss_name == 'CE':
            return CELoss(kwargs.get('label_smoothing', 0.0))
        elif loss_name == 'Focal':
            return FocalLoss(kwargs.get('alpha_val', 0.25), kwargs.get('gamma_val', 2.0))
        elif loss_name == 'swin_loss':
            return SwinCombinedLoss()
        else:
            raise ValueError(f"'{loss_name}' is not a valid loss name")
