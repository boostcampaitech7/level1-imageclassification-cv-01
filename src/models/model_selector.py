import torch
import torch.nn as nn
from torchvision import models
import timm

from . import SimpleCNN
from . import CLIP_backbone
from . import CNNViTModel
from . import EnsembleModel
from . import ModifiedSwinTransformer
from . import Coca_backbone
import math

import torch
import torch.nn as nn


# def build_model(model_name='base',**kwargs):
#     if model_name == 'mnist_linear':
#         return SimpleCNN(**kwargs)
#     elif model_name == 'mnist_conv':
#         return MNIST_conv(**kwargs)
#     else:
#         raise ValueError('not a correct model name', model_name)


class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """

    def __init__(
        self, 
        model_type: str, 
        model_name,
        num_classes: int,
        pretrained=False,
        num_cnn_classes=None, 
        **kwargs
    ):

        # 모델 유형에 따라 적절한 모델 객체를 생성
        if model_type == "simple":
            self.model = SimpleCNN(num_classes=num_classes)

        elif model_type == "torchvision":
            self.model = TorchvisionModel(model_name, num_classes, pretrained)

        elif model_type == "timm":
            self.model = TimmModel(model_name, num_classes, pretrained)
        
        elif model_type == "clip":
            self.model = CLIP_backbone(model_name,num_classes)

        elif model_type == 'openclip':
            self.model = Coca_backbone(model_name,num_classes)

        elif model_type == 'cnnvit':
            self.model = CNNViTModel(num_cnn_classes, num_classes, pretrained)
        
        elif model_type == 'ensemble':
            self.model = EnsembleModel(num_classes=num_classes)

        elif model_type == 'swin':
            self.model = ModifiedSwinTransformer(num_classes, num_classes_large=8, num_classes_small=154)
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:
        return self.model


class TorchvisionModel(nn.Module):
    """
    Torchvision에서 제공하는 사전 훈련된 모델을 사용하는 클래스.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool):
        super(TorchvisionModel, self).__init__()
        self.model = models.__dict__[model_name](pretrained=pretrained)

        # 모델의 최종 분류기 부분을 사용자 정의 클래스 수에 맞게 조정
        if "fc" in dir(self.model):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)

        elif "classifier" in dir(self.model):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)