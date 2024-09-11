import torch
from torch.nn import functional as F
import torch.nn as nn

import pytorch_lightning as pl
import torchmetrics

import net
from losses import get_loss


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
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc,on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss

    # def training_epoch_end(self, outputs):
    #     return None


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        

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
            return torch.optim.Adam(self.backbone.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('not a correct optim name', self.optim)
