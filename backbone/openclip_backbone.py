import timm
import torch
from torch import nn 
import open_clip


class Coca_backbone(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """

    def __init__(self, model_name: str, num_classes: int):
        super(Coca_backbone, self).__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained=True, num_classes=num_classes
        # )
        
        self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_name)



        for param in self.model.parameters():
            param.requires_grad = False

        # for param in self.model.head.parameters():
        #     param.requires_grad = True
        
        self.fc = nn.Linear(self.model.ln_final.normalized_shape[0],num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.fc(self.model.encode_image(x))
