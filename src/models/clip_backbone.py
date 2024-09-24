import timm
import torch
from torch import nn 

class CLIP_backbone(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """

    def __init__(self, model_name: str, num_classes: int):
        super(CLIP_backbone, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        
        # if 'clip' not in model_name:
        #     raise ValueError("give me clip model")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.head.parameters():
            param.requires_grad = True
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)
