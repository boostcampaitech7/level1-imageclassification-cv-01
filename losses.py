import torch
import torch.nn as nn

def get_loss(loss_name='CE',**kwargs):
    if loss_name == 'CE':
        return CELoss()
    else:
        raise ValueError('not a correct model name', loss_name)


class CELoss(nn.Module):
    """
    모델의 손실함수를 계산하는 클래스.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)