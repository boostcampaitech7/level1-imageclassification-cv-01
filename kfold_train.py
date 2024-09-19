import os
from argparse import ArgumentParser
from time import gmtime, strftime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pl_trainer import Sketch_Classifier
from utils.util import dotdict

from sklearn.model_selection import StratifiedKFold

import pandas as pd

from data_sets import base_dataset
from select_transforms import TransformSelector


# TODO
# test option 따로 만들기
# model ckpt 파일 불러오기
# pin mem등 pl에서 정하는 가속 설정 
# https://lightning.ai/docs/pytorch/stable/advanced/speed.html
# wandb team space 생성
# data module 분리 -> validataion data 분리, 전처리가 상이할 수 있음 



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args(config):
    parser = ArgumentParser()

    # Set defaults from config file, but allow override via command line
    parser.add_argument('--exp_name', type=str, default=config.get('exp_name')) # 현재 실험 이름 
    parser.add_argument('--base_output_dir', type=str, default=config.get('base_output_dir')) # 실험 결과 저장 폴더 
    parser.add_argument('--gpus', type=str, default=config.get('gpus'))
    
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size'))
    parser.add_argument('--epochs', type=int, default=config.get('epochs'))
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate'))

    parser.add_argument('--model_type', type=str, default=config.get('model_type')) # backbone 타입
    parser.add_argument('--model_name', type=str, default=config.get('model_name')) # torchvision, timm을 위한 model 이름 
    parser.add_argument('--pretrained', type=bool, default=config.get('pretrained'))
    parser.add_argument('--data_name', type=str, default=config.get('data_name')) # dataset 이름
    parser.add_argument('--transform_name', type=str, default=config.get('transform_name')) 
    parser.add_argument('--num_classes', type=int, default=config.get('num_classes'))
    parser.add_argument('--optim', type=str, default=config.get('optim'))
    parser.add_argument('--weight_decay', type=str, default=config.get('weight_decay'))
    parser.add_argument('--loss', type=str, default=config.get('loss'))
    parser.add_argument('--cos_sch', type=int, default=config.get('cos_sch')) # cos 주기
    parser.add_argument('--warm_up', type=int, default=config.get('warm_up'))
    parser.add_argument('--early_stopping', type=int , default=config.get('early_stopping'))

    parser.add_argument('--train_data_dir', type=str, default=config.get('train_data_dir'))
    parser.add_argument('--traindata_info_file', type=str, default=config.get('traindata_info_file'))
    parser.add_argument('--test_data_dir', type=str, default=config.get('test_data_dir'))
    parser.add_argument('--testdata_info_file', type=str, default=config.get('testdata_info_file'))
    
    parser.add_argument('--use_wandb', type=int, default=config.get('use_wandb')) # wandb 사용?
    parser.add_argument('--num_workers', type=str, default=config.get('num_workers')) # dataloader 옵션 관련 
    # 경윤---
    parser.add_argument('--cutmix_mixup', type=str, default=config.get('cutmix_mixup'))
    # 경윤---
    parser.add_argument('--kfold_pl_train_return', type=str, default=True)
    return parser.parse_args()



def main(args):
    
    hparams = dotdict(vars(args))
    # ------------
    # data
    # ------------
    

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_info_df = pd.read_csv(hparams['traindata_info_file'])

    transform_selector = TransformSelector(hparams.transform_name)
    train_transform = transform_selector.get_transform(True)
    test_transform = transform_selector.get_transform(False)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_info_df, train_info_df['target'])):
        print('fold:',fold)
        train_df = train_info_df.iloc[train_idx]
        val_df = train_info_df.iloc[val_idx]

        train_dataset = base_dataset.CustomDataset(hparams.train_data_dir,train_df,train_transform,False)
        val_dataset = base_dataset.CustomDataset(hparams.train_data_dir,val_df,test_transform,False)
        train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=False)

        # logger
        csv_logger = CSVLogger(save_dir=hparams.output_dir+f'_fold{fold}', name='result')
        my_loggers = [csv_logger]

        # create model checkpoint callback
        monitor = 'val_acc'
        mode = 'max'
        save_top_k = 1 # best 하나 저장 + Last 저장 
        checkpoint_callback = [] 
        checkpoint_callback.append(ModelCheckpoint(dirpath=hparams.output_dir+f'_fold{fold}', save_last=True,
                                            save_top_k=save_top_k, monitor=monitor, mode=mode))
        
        
        if hparams.early_stopping>0:
            early_stop = EarlyStopping(monitor="valid_loss", patience=hparams.early_stopping, verbose=False, mode="min")
            checkpoint_callback.append(early_stop)

        # ------------
        # training
        # ------------
        trainer = pl.Trainer(logger=my_loggers,
                            accelerator='cpu' if hparams.gpus == 0 else 'gpu',
                            devices=None if hparams.gpus == 0 else hparams.gpus,
                            callbacks=checkpoint_callback,
                            max_epochs=hparams.epochs)
        
        trainer_mod = Sketch_Classifier(**hparams)
        trainer.fit(trainer_mod, train_loader,val_loader)
        # ------------
        # testing
        # ------------

        best_model_path = checkpoint_callback[0].best_model_path

        best_model = Sketch_Classifier.load_from_checkpoint(best_model_path, **hparams)

        
        print('start predict validation dataset')

        # data_mod.test_data_dir = data_mod.train_data_dir
        # data_mod.test_info_df = data_mod.val_info_df
        # data_mod.setup(stage='predict')

        val_dataset = base_dataset.CustomDataset(hparams.train_data_dir,val_df,test_transform,True)
        val_loader = DataLoader(val_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=False)

        validations = trainer.predict(best_model, dataloaders=val_loader)
        
        # pred_list = [pred.cpu().numpy() for batch in validations for pred, logit in batch]
        # logit_list = [logit.cpu().numpy() for batch in validations for pred, logit in batch]

        pred_list = []
        logit_list = []

        for batch in validations:
            preds, logits = batch  # batch에서 preds와 logits를 가져옴
            pred_list.extend(preds.cpu().numpy())  # 각 배치의 pred들을 리스트에 추가
            logit_list.extend(logits.cpu().numpy())  # 각 배치의 logit들을 리스트에 추가

        # val_info = val_df.iloc[:100]
        val_info = val_df
        val_info['pred'] = pred_list
        val_info['logit'] = [logit.tolist() for logit in logit_list]
        val_info.to_csv(os.path.join(args.output_dir+f'_fold{fold}',f"{fold}_validation.csv"), index=False)

if __name__ == '__main__':

    pl.seed_everything(42)
    
    # ------------
    # args
    # ------------

    config = load_config('config.yaml')
    args = parse_args(config)

    ## output_dir 
    current_time = strftime("%m-%d_0", gmtime())
    pt = "O" if args.pretrained else 'X'
    name_str =  f"{args.model_name}-{args.batch_size}-{args.learning_rate}"+\
                f"-{args.optim}-{pt}-{args.exp_name}"
    # args.output_dir = os.path.join(args.base_output_dir, args.exp_name + "_" + current_time)
    args.output_dir = os.path.join(args.base_output_dir, name_str + "_" + current_time)
    if os.path.isdir(args.output_dir):
        while True:
            cur_exp_number = int(args.output_dir[-2:].replace('_', ""))
            args.output_dir = args.output_dir[:-2] + "_{}".format(cur_exp_number+1)
            if not os.path.isdir(args.output_dir):
                break

    # gpus
    args.gpus = [int(i) for i in str(args.gpus).split(',')]

    main(args)