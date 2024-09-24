import torch
import torch.nn as nn
from torchvision import models
import timm

from backbone.base_backbone import SimpleCNN
from backbone.clip_backbone import CLIP_backbone
from backbone.openclip_backbone import Coca_backbone
from backbone.swin_backbone import ModifiedSwinTransformer
import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vit_b_16

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
    

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class CNNViTModel(nn.Module):
    def __init__(self, num_cnn_classes, num_classes, pretrained):
        super(CNNViTModel, self).__init__()
        
        # CNN 부분 (ResNet50, resnext50_32x4d 등)
        self.cnn_model = models.resnext50_32x4d(pretrained=pretrained)
        self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, num_cnn_classes)

        self.pos_embed = self.create_sincos_positional_embedding(
            (224 // 16)**2 + 1, 768
        )
        self.patch_embed = PatchEmbedding()

        self.vit = vit_b_16(pretrained=pretrained)
        self.vit_encoder = self.vit.encoder

        self.linear_layer = nn.Linear(num_cnn_classes, 768)

        self.vit.heads = nn.Linear(768, num_classes)

    def create_sincos_positional_embedding(self, n_positions, dim):
        position = torch.arange(n_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(n_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # CNN
        cnn_logits = self.cnn_model(x)
        
        # cls_token_embedding
        cls_token_embedding = self.linear_layer(cnn_logits)

        # PatchEmbedding
        vit_input = self.patch_embed(x)

        cls_token = cls_token_embedding.unsqueeze(1)
        vit_input = torch.cat([cls_token, vit_input], dim=1)
        vit_input += self.pos_embed

        out = self.vit_encoder(vit_input)

        logits = self.vit.heads(out[:, 0])

        return logits