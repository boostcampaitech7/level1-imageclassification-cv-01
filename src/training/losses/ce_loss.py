import torch
import torch.nn as nn

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