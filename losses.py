import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(loss_name='CE',**kwargs):
    if loss_name == 'CE':
        return CELoss(kwargs.get('label_smoothing',0.0))
    elif loss_name == 'Focal':
        return FocalLoss(kwargs.get('alpha_val',0.25),kwargs.get('gamma_val',2.0))
    elif loss_name == 'swin_loss':
        return CombinedLoss()
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

        
class CombinedLoss(nn.Module):
    def __init__(self, large_loss_weight=0.4, small_loss_weight=0.2, original_loss_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.large_loss_fn = nn.CrossEntropyLoss()
        self.small_loss_fn = nn.CrossEntropyLoss()
        self.original_loss_fn = nn.CrossEntropyLoss()
        self.large_loss_weight = large_loss_weight
        self.small_loss_weight = small_loss_weight
        self.original_loss_weight = original_loss_weight

    def forward(self, original_output, original_label, large_output, large_label, small_output, small_label):

        original_loss = self.original_loss_fn(original_output, original_label)
        large_loss = self.large_loss_fn(large_output, large_label)
        small_loss = self.small_loss_fn(small_output, small_label)

        total_loss = (
            (self.large_loss_weight * large_loss) + 
            (self.small_loss_weight * small_loss) +
            (self.original_loss_weight * original_loss)
        )
        return total_loss