import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import f1_score

import pytorch_lightning as pl
import torchmetrics

import net
from losses import get_loss

import itertools
import numpy as np
import pandas as pd 
import os 


def cutmix(batch, alpha=0.9):
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

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets

class Sketch_Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model_select = net.ModelSelector(kwargs['model_type'],kwargs['model_name'],kwargs['num_classes'],kwargs['pretrained'])

        # self.backbone = net.build_model(model_name = kwargs['arch'])
        self.backbone = self.model_select.get_model()
        self.learning_rate = kwargs['learning_rate'] 

        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=kwargs['num_classes'])
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = get_loss(loss_name=kwargs['loss'],**kwargs)
        self.optim = kwargs['optim']
        self.weight_decay = kwargs['weight_decay']
        self.cos_sch = kwargs['cos_sch']
        self.warm_up = kwargs['warm_up']
        self.output_dir = kwargs['output_dir']


    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # 기존 배치에서 x, y를 가져오기
        x, y = batch
        
        # CutMix를 적용한 새로운 데이터 생성
        x_cutmix, (y1, y2, lam) = cutmix(batch, alpha=0.9)

        # 모델 예측 (원본 데이터)
        y_hat_original = self(x)
        
        # 모델 예측 (CutMix된 데이터)
        y_hat_cutmix = self(x_cutmix)
        
        # Loss 계산 (원본 데이터에 대한 손실)
        loss_original = self.criterion(y_hat_original, y)
        
        # Loss 계산 (CutMix 데이터에 대한 손실)
        loss_cutmix = lam * self.criterion(y_hat_cutmix, y1) + (1 - lam) * self.criterion(y_hat_cutmix, y2)
        
        # 두 손실을 합산
        total_loss = loss_original + loss_cutmix

        # 예측 결과 합치기
        y_combined = torch.cat([y, y1], dim=0)  # 원본 데이터의 타겟과 CutMix의 타겟을 결합
        y_hat_combined = torch.cat([y_hat_original, y_hat_cutmix], dim=0)  # 원본 예측과 CutMix 예측을 결합
        
        # 정확도 계산 (원본과 CutMix 데이터 모두 포함)
        total_acc = self.accuracy(y_hat_combined, y_combined)
        
        #lr log 추가 
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)

        self.log('train_loss', total_loss, on_epoch=True)
        self.log('train_acc', total_acc,on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return total_loss

    # def training_epoch_end(self, outputs):
    #     return None


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)  
        acc  = self.accuracy(y_hat, y)
        f1 = f1_score(y.cpu(), preds.cpu(), average='macro')  # 'macro'는 클래스에 대한 평균을 의미합니다.
        
        self.log('valid_loss', loss, on_step=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)



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

        return preds


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        
        if self.optim == 'Adam':
            opt = torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'AdamW':
            opt = torch.optim.AdamW(self.backbone.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'RAdam':
            opt = torch.optim.RAdam(self.backbone.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError('not a correct optim name', self.optim)
        
        schedulers = []
        milestones = []

        if self.warm_up>0:
            #TODO
            warm_up = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=self.warm_up)
            schedulers.append(warm_up)
            milestones.append(self.warm_up)  # warmup 끝나는 시점 설정 (10 에포크)
        
        if self.cos_sch>0:

            cos_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cos_sch)
            schedulers.append(cos_sch)

        
        if len(schedulers)>0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt,schedulers=schedulers,milestones=milestones[:len(schedulers)-1])
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "step", # step means "batch" here, default: epoch   # New!
                    "frequency": 1, # default
                },
            }
        else:
            return opt 