import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(loss_name='CE',**kwargs):
    if loss_name == 'CE':
        return CELoss(kwargs.get('label_smoothing',0.0))
    elif loss_name == 'Focal':
        return FocalLoss(kwargs.get('alpha_val',0.25),kwargs.get('gamma_val',2.0))
    else:
        raise ValueError('not a correct model name', loss_name)

class CELoss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self,label_smoothing):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing = label_smoothing)

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss