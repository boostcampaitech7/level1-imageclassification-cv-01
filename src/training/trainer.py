import itertools
import os

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from sklearn.metrics import f1_score

from ..models import model_selector
from .losses import LossSelector


def cutmix(batch, alpha=0.9, apply_ratio=1.0):
    
    data, targets = batch
    # 디버깅: targets에 빈 값이 있는지 확인
    if targets is None or len(targets) == 0:
        raise ValueError("targets가 비어 있습니다.")
    
    batch_size = data.size(0)
    num_apply = max(1, int(batch_size * apply_ratio))
    apply_indices = torch.randperm(batch_size)[:num_apply]
        
    cutmix_data = data.clone()
    cutmix_targets = []
    
    for i in apply_indices:
        
        # 현재 이미지와 다른 이미지를 선택
        while True:
            j = torch.randint(0, batch_size, (1,)).item()
            if j != i:
                break
        
        lam = np.random.uniform(0.1, alpha)

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        cutmix_data[i, :, y0:y1, x0:x1] = data[j, :, y0:y1, x0:x1]
        cutmix_targets.append((targets[i], targets[j], lam))

    return cutmix_data[apply_indices], cutmix_targets


def mixup(images, labels, alpha=1.0, apply_ratio = 1.0):
    
    batch_size = len(images)
    num_apply = max(1,int(batch_size * apply_ratio))

    # 적용할 이미지 인덱스를 무작위로 선택
    apply_indices = torch.randperm(batch_size)[:num_apply]

    mixedup_images = []
    mixedup_labels = []
    lam_list = []


    for i in apply_indices:
        # 현재 이미지와 다른 이미지를 선택
        while True:
            j = torch.randint(0, batch_size, (1,)).item()
            if j != i:
                break
        
        # 베타 분포에서 샘플링된 값으로 lambda를 구함
        lam = np.clip(np.random.beta(alpha, alpha), 0.5, 0.6)
        
        # 이미지와 레이블을 lam 비율에 따라 선형 결합
        mixed_image = lam * images[i] + (1 - lam) * images[j]
        mixed_label = lam * labels[i] + (1 - lam) * labels[j]
        
        mixedup_images.append(mixed_image)
        mixedup_labels.append(mixed_label)
        lam_list.append(lam)
    
    # 리스트를 텐서로 변환
    mixedup_images = torch.stack(mixedup_images)
    mixedup_labels = torch.stack(mixedup_labels)    
    
    return mixedup_images, mixedup_labels, lam_list


class Sketch_Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_select = model_selector.ModelSelector(
            kwargs["model_type"],
            kwargs["model_name"],
            kwargs["num_classes"],
            kwargs["pretrained"],
            kwargs["num_cnn_classes"],
        )

        # self.backbone = net.build_model(model_name = kwargs['arch'])
        self.backbone = self.model_select.get_model()
        self.learning_rate = kwargs["learning_rate"]

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=kwargs["num_classes"]
        )
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = LossSelector(loss_name=kwargs["loss"], **kwargs)
        self.optim = kwargs["optim"]
        self.weight_decay = kwargs["weight_decay"]
        self.cos_sch = kwargs["cos_sch"]
        self.warm_up = kwargs["warm_up"]
        self.output_dir = kwargs["output_dir"]

        self.k_fold_option = kwargs["kfold_pl_train_return"]

        self.num_classes = kwargs["num_classes"]
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.cutmix_mixup = kwargs["cutmix_mixup"]
        self.cutmix_ratio = kwargs["cutmix_ratio"]
        self.mixup_ratio = kwargs["mixup_ratio"]
        self.model_type = kwargs["model_type"]

    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        if self.model_type == 'swin':
            x, y, large_label, small_label = batch
            y_hat_original, large_output, small_output = self(x)
            loss_original = self.criterion(y_hat_original, y, large_output, large_label, small_output, small_label)
        else:
            x, y = batch
            y_hat_original = self(x)
            loss_original = self.criterion(y_hat_original, y)
        
        total_acc = self.accuracy(y_hat_original, y)

        if self.cutmix_mixup == "cutmix":
            
            # CutMix를 적용한 새로운 데이터 생성
            x_cutmix, new_targets = cutmix(batch, alpha=0.9, apply_ratio=self.cutmix_ratio)
            
            # 모델 예측 (CutMix된 데이터)
            y_hat_cutmix = self(x_cutmix)
            
            # Loss 계산 (CutMix 데이터에 대한 손실)
            loss_cutmix = 0
            for i, target in enumerate(new_targets):
                if isinstance(target, tuple):  # CutMix가 적용된 이미지
                    y1, y2, lam = target
                    loss_cutmix += lam * self.criterion(y_hat_cutmix[i].unsqueeze(0), y1.unsqueeze(0)) + \
                                (1 - lam) * self.criterion(y_hat_cutmix[i].unsqueeze(0), y2.unsqueeze(0))
                
            loss_cutmix = loss_cutmix / len(new_targets)           
            # 두 손실을 합산
            total_loss = 0.6 * loss_original + 0.4 * loss_cutmix

        elif self.cutmix_mixup == "mixup":

            # 라벨을 원-핫 인코딩으로 변환 (Mixup에만 사용)
            y_onehot = F.one_hot(y, self.num_classes).float()
            
            # Mixup 적용
            x_mixup, y_mixup, lam_mixup = mixup(x, y_onehot, alpha=0.8, apply_ratio=self.mixup_ratio)

            # Mixup 데이터에 대한 예측
            y_hat_mixup = self(x_mixup)


            # Mixup 데이터에 대한 손실 계산
            loss_mixup = 0
            for i, (y_mix, lam) in enumerate(zip(y_mixup, lam_mixup)):
                loss_mixup += lam * self.criterion_bce(y_hat_mixup[i].unsqueeze(0), y_onehot[i].unsqueeze(0)) + \
                            (1 - lam) * self.criterion_bce(y_hat_mixup[i].unsqueeze(0), y_mix.unsqueeze(0))
            
            # 평균 손실 계산
            loss_mixup /= len(y_mixup)

            # 총 손실 합산
            total_loss = 0.6 * loss_original + 0.4 * loss_mixup

        elif self.cutmix_mixup == "cutmix_mixup" or self.cutmix_mixup == "mixup_cutmix":

            # CutMix
            # CutMix를 적용한 새로운 데이터 생성
            x_cutmix, new_targets = cutmix(batch, alpha=0.9, apply_ratio=self.cutmix_ratio)
            
            # 모델 예측 (CutMix된 데이터)
            y_hat_cutmix = self(x_cutmix)
            
            # Loss 계산 (CutMix 데이터에 대한 손실)
            loss_cutmix = 0
            for i, target in enumerate(new_targets):
                if isinstance(target, tuple):  # CutMix가 적용된 이미지
                    y1, y2, lam = target
                    loss_cutmix += lam * self.criterion(y_hat_cutmix[i].unsqueeze(0), y1.unsqueeze(0)) + \
                                (1 - lam) * self.criterion(y_hat_cutmix[i].unsqueeze(0), y2.unsqueeze(0))
            
 
            loss_cutmix = loss_cutmix / len(new_targets)       
                       
            # Mixup
            # 라벨을 원-핫 인코딩으로 변환 (Mixup에만 사용)
            y_onehot = F.one_hot(y, self.num_classes).float()
            
            # Mixup 적용
            x_mixup, y_mixup, lam_mixup = mixup(x, y_onehot, alpha=0.8, apply_ratio=self.mixup_ratio)

            # Mixup 데이터에 대한 예측
            y_hat_mixup = self(x_mixup)


            # Mixup 데이터에 대한 손실 계산
            loss_mixup = 0
            for i, (y_mix, lam) in enumerate(zip(y_mixup, lam_mixup)):
                loss_mixup += lam * self.criterion_bce(y_hat_mixup[i].unsqueeze(0), y_onehot[i].unsqueeze(0)) + \
                            (1 - lam) * self.criterion_bce(y_hat_mixup[i].unsqueeze(0), y_mix.unsqueeze(0))
            
            # 평균 손실 계산
            loss_mixup /= len(y_mixup)

            total_loss = 0.4 * loss_original + 0.3 * loss_cutmix + 0.3 * loss_mixup

        else:
            total_loss = loss_original

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=True, on_epoch=False)

        self.log("train_loss", total_loss, on_epoch=True)
        self.log(
            "train_acc",
            total_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    # def training_epoch_end(self, outputs):
    #     return None

    def validation_step(self, batch, batch_idx):
        if self.model_type == 'swin':
            x, y, large_label, small_label = batch
            y_hat, large_output, small_output = self(x)
            loss = self.criterion(y_hat, y, large_output, large_label, small_output, small_label)

        else:
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(y_hat, y)
        f1 = f1_score(
            y.cpu(), preds.cpu(), average="macro"
        )  # 'macro'는 클래스에 대한 평균을 의미합니다.

        self.log("valid_loss", loss, on_step=True)
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.criterion(y_hat, y)

    #     acc = self.accuracy(y_hat, y)
    #     self.log('test_loss', loss)
    #     self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        if self.model_type == 'swin':
            x, _, _ = self(batch)
        else:
            x = self(batch)
        logits = F.softmax(x, dim=1)
        preds = logits.argmax(dim=1)

        if self.k_fold_option:
            return preds, logits
        else:
            return preds

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()

        if self.optim == "Adam":
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.backbone.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "AdamW":
            opt = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.backbone.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "RAdam":
            opt = torch.optim.RAdam(
                filter(lambda p: p.requires_grad, self.backbone.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "SGD":
            opt = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.backbone.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError("not a correct optim name", self.optim)

        schedulers = []
        milestones = []

        if self.warm_up > 0:
            # TODO
            warm_up = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=self.warm_up
            )
            schedulers.append(warm_up)
            milestones.append(self.warm_up)  # warmup 끝나는 시점 설정 (10 에포크)

        if self.cos_sch > 0:

            cos_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.cos_sch
            )
            schedulers.append(cos_sch)

        if len(schedulers) > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=schedulers, milestones=milestones[: len(schedulers) - 1]
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "step",  # step means "batch" here, default: epoch   # New!
                    "frequency": 1,  # default
                },
            }
        else:
            return opt
