import torch
from torch.nn import functional as F
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

import net
from losses import get_loss

import itertools
import numpy as np
import pandas as pd 
import os 


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
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        acc  = self.accuracy(y_hat, y)
        # https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        
        #lr log 추가 
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc,on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    # def training_epoch_end(self, outputs):
    #     return None


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # preds = torch.argmax(y_hat, dim=1)  

        acc  = self.accuracy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)



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
            opt = torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate,weight_decay = self.weight_decay)
            
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