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

import net
from losses import get_loss


def cutmix(batch, alpha=0.9, apply_ratio=1.0):

    # 확률적으로 cutmix 적용
    if np.random.rand() > apply_ratio:
        return batch  # cutmix를 적용하지 않음

    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
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

    # inplace 연산 역전파 오류 방지
    new_data = data.clone()  # 원본 데이터를 클론하여 새로운 텐서를 생성
    new_data[:, :, y0:y1, x0:x1] = shuffled_data[
        :, :, y0:y1, x0:x1
    ]  # in-place 연산 방지
    targets = (targets, shuffled_targets, lam)

    return new_data, targets


def mixup(images, labels, alpha=1.0, apply_ratio=1.0):
    # 확률적으로 mixup을 적용할지를 결정
    if np.random.rand() > apply_ratio:
        return images, labels, None  # mixup을 적용하지 않음

    # 배치 내 이미지와 레이블의 순서를 무작위로 섞음
    indices = torch.randperm(len(images))
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    # 베타 분포에서 샘플링된 값으로 lambda를 구함
    lam = np.clip(np.random.beta(alpha, alpha), 0.5, 0.6)

    # 이미지와 레이블을 lam 비율에 따라 선형 결합
    mixedup_images = lam * images + (1 - lam) * shuffled_images
    mixedup_labels = lam * labels + (1 - lam) * shuffled_labels

    return mixedup_images, mixedup_labels, lam


class Sketch_Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_select = net.ModelSelector(
            kwargs["model_type"],
            kwargs["model_name"],
            kwargs["num_classes"],
            kwargs["pretrained"],
        )

        # self.backbone = net.build_model(model_name = kwargs['arch'])
        self.backbone = self.model_select.get_model()
        self.learning_rate = kwargs["learning_rate"]

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=kwargs["num_classes"]
        )
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = get_loss(loss_name=kwargs["loss"], **kwargs)
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

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat_original = self(x)
        loss_original = self.criterion(y_hat_original, y)
        total_acc = self.accuracy(y_hat_original, y)

        if self.cutmix_mixup == "cutmix":

            # CutMix를 적용한 새로운 데이터 생성
            x_cutmix, (y1, y2, lam) = cutmix(
                batch, alpha=0.9, apply_ratio=self.cutmix_ratio
            )

            # 모델 예측 (CutMix된 데이터)
            y_hat_cutmix = self(x_cutmix)

            # Loss 계산 (CutMix 데이터에 대한 손실)
            loss_cutmix = lam * self.criterion(y_hat_cutmix, y1) + (
                1 - lam
            ) * self.criterion(y_hat_cutmix, y2)

            # 두 손실을 합산
            total_loss = 0.6 * loss_original + 0.4 * loss_cutmix

        elif self.cutmix_mixup == "mixup":

            # 라벨을 원-핫 인코딩으로 변환 (Mixup에만 사용)
            y_onehot = F.one_hot(y, self.num_classes).float()

            # Mixup 적용
            x_mixup, y_mixup, lam_mixup = mixup(
                x, y_onehot, alpha=0.8, apply_ratio=self.mixup_ratio
            )

            # Mixup 데이터에 대한 예측
            y_hat_mixup = self(x_mixup)

            # Mixup 데이터에 대한 손실
            loss_mixup = lam_mixup * self.criterion_bce(y_hat_mixup, y_onehot) + (
                1 - lam_mixup
            ) * self.criterion_bce(y_hat_mixup, y_mixup)

            # 총 손실 합산
            total_loss = 0.6 * loss_original + 0.4 * loss_mixup

        elif self.cutmix_mixup == "cutmix_mixup" or self.cutmix_mixup == "mixup_cutmix":

            # CutMix
            x_cutmix, (y1, y2, lam) = cutmix(batch, alpha=0.9)
            y_hat_cutmix = self(x_cutmix)
            loss_cutmix = lam * self.criterion(y_hat_cutmix, y1) + (
                1 - lam
            ) * self.criterion(y_hat_cutmix, y2)

            # Mixup
            y_onehot = F.one_hot(y, self.num_classes).float()
            x_mixup, y_mixup, lam_mixup = mixup(x, y_onehot, alpha=0.8)
            y_hat_mixup = self(x_mixup)
            loss_mixup = lam_mixup * self.criterion_bce(y_hat_mixup, y_onehot) + (
                1 - lam_mixup
            ) * self.criterion_bce(y_hat_mixup, y_mixup)

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
                self.backbone.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "AdamW":
            opt = torch.optim.AdamW(
                self.backbone.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "RAdam":
            opt = torch.optim.RAdam(
                self.backbone.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim == "SGD":
            opt = torch.optim.SGD(
                self.backbone.parameters(),
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
