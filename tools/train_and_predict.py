import os
from argparse import ArgumentParser
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import yaml
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
import torch.nn.functional as F

from src.data import data_module, base_dataset
from src.data import TransformSelector
from src.training import Sketch_Classifier
from src.utils import dotdict, load_config, setup_logger

# TODO
# test option 따로 만들기
# model ckpt 파일 불러오기
# pin mem등 pl에서 정하는 가속 설정
# https://lightning.ai/docs/pytorch/stable/advanced/speed.html
# wandb team space 생성
# data module 분리 -> validataion data 분리, 전처리가 상이할 수 있음


def parse_args(config):
    parser = ArgumentParser()

    # Set defaults from config file, but allow override via command line
    parser.add_argument("--exp_name", type=str, default=config.get("exp_name"))
    parser.add_argument("--base_output_dir", type=str, default=config.get("base_output_dir"))
    parser.add_argument("--gpus", type=str, default=config.get("gpus"))
    parser.add_argument("--batch_size", type=int, default=config.get("batch_size"))
    parser.add_argument("--epochs", type=int, default=config.get("epochs"))
    parser.add_argument("--learning_rate", type=float, default=config.get("learning_rate"))
    parser.add_argument("--num_cnn_classes", type=int, default=config.get("num_cnn_classes"))
    parser.add_argument("--model_type", type=str, default=config.get("model_type"))
    parser.add_argument("--model_name", type=str, default=config.get("model_name"))
    parser.add_argument("--pretrained", type=bool, default=config.get("pretrained"))
    parser.add_argument("--data_name", type=str, default=config.get("data_name"))
    parser.add_argument("--transform_name", type=str, default=config.get("transform_name"))
    parser.add_argument("--num_classes", type=int, default=config.get("num_classes"))
    parser.add_argument("--optim", type=str, default=config.get("optim"))
    parser.add_argument("--weight_decay", type=str, default=config.get("weight_decay"))
    parser.add_argument("--loss", type=str, default=config.get("loss"))
    parser.add_argument("--cos_sch", type=int, default=config.get("cos_sch"))
    parser.add_argument("--warm_up", type=int, default=config.get("warm_up"))
    parser.add_argument("--early_stopping", type=int, default=config.get("early_stopping"))
    parser.add_argument("--train_data_dir", type=str, default=config.get("train_data_dir"))
    parser.add_argument("--traindata_info_file", type=str, default=config.get("traindata_info_file"))
    parser.add_argument("--test_data_dir", type=str, default=config.get("test_data_dir"))
    parser.add_argument("--testdata_info_file", type=str, default=config.get("testdata_info_file"))
    parser.add_argument("--use_wandb", type=int, default=config.get("use_wandb"))
    parser.add_argument("--num_workers", type=str, default=config.get("num_workers"))
    parser.add_argument("--cutmix_mixup", type=str, default=config.get("cutmix_mixup"))
    parser.add_argument("--cutmix_ratio", type=int, default=config.get("cutmix_ratio"))
    parser.add_argument("--mixup_ratio", type=int, default=config.get("mixup_ratio"))
    parser.add_argument("--kfold_pl_train_return", type=str, default=False)
    parser.add_argument("--n_splits", type=int, default=config.get("n_splits"))
    parser.add_argument("--sweep_mode", type=bool, default=config.get("sweep_mode"))

    parser.add_argument(
        "--mixed_precision", type=bool, default=config.get("mixed_precision"),
        help="Use mixed precision training for better performance."
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=config.get("accumulate_grad_batches"),
        help="Accumulate gradients over multiple batches to save memory."
    )
    parser.add_argument(
        "--use_kfold", type=bool, default=config.get("use_kfold"),
        help="Whether to use K-Fold cross-validation during training."
    )

    return parser.parse_args()

def run_sweep():
    sweep_config = {
        "name": "ResnextTesting",
        "method": "bayes", # random -> bayes 
        "metric": {"goal": "minimize", "name": "valid_loss_epoch"},
        "parameters": {
            "learning_rate": {"min": 0.00001, "max": 0.001},
            "batch_size": {"values": [32, 64, 128]},
            "optimizer": {"values": ["AdamW", "SGD"]},
            "loss": {"values": ["Focal", "CE"]},
            "label_smoothing": {
                "values": [0.0, 0.1, 0.2, 0.3],
                #"condition": {"loss": "CE"}  # 조건 설정
            },
            # "focal_constant": {
            #     "alpha_val": [0.1, 0.25, 0.5, 0.75, 1.0],
            #     "gamma_val": [0.0, 1.0, 2.0, 3.0, 5.0],
            #     "condition": {"loss": "focal"}
            # },
            "focal_alpha": {
                "values": [0.1, 0.25, 0.5, 0.75, 1.0],
                #"condition": {"loss": "Focal"}  # 조건 설정
            },
            "focal_gamma": {
                "values":[0.0, 1.0, 2.0, 3.0, 5.0],
                #"condition": {"loss": "Focal"}
            },
            "cutmix_ratio": {"min": 0.1, "max": 0.4},
            "mixup_ratio": {"min": 0.1, "max": 0.4},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="sketch classification", entity="nav_sketch")
    wandb.agent(sweep_id, function=lambda: main(args), count=30) # 각 모델에 대해 한 번씩 실행



def train(trainer_mod, data_mod,my_loggers,checkpoint_callback,**hparams):
    trainer = pl.Trainer(
        logger=my_loggers,
        accelerator="cpu" if hparams['gpus'] == 0 else "gpu",
        precision=16 if hparams['gpus'] != 0 else 32,  # CPU에서는 32-bit precision
        devices=None if hparams['gpus'] == 0 else hparams['gpus'],
        callbacks=checkpoint_callback,  # 콜백 리스트로 묶는 것이 좋음
        max_epochs=hparams['epochs'],
        accumulate_grad_batches=(1 if hparams['accumulate_grad_batches'] <= 0
            else hparams['accumulate_grad_batches']),
    )

    
    trainer.fit(trainer_mod, data_mod)

    return trainer,trainer_mod,data_mod


def val_pred(trainer,trainer_mod,data_mod,**hparams):

    data_mod.test_data_dir = data_mod.train_data_dir
    data_mod.test_info_df = data_mod.val_info_df
    data_mod.setup(stage="predict")

    validations = trainer.predict(trainer_mod, dataloaders=data_mod.predict_dataloader())

    # pred_list = [pred.cpu().numpy() for batch in validations for pred, logit in batch]
    # logit_list = [logit.cpu().numpy() for batch in validations for pred, logit in batch]

    pred_list = []
    # if hparams['kfold_pl_train_return']:
    #     logit_list = []

    for batch in validations:
        if hparams['kfold_pl_train_return']:
            preds, logits = batch  # batch에서 preds와 logits를 가져옴
        else:
            preds = batch 
        pred_list.extend(preds.cpu().numpy())  # 각 배치의 pred들을 리스트에 추가
        # if hparams['kfold_pl_train_return']:
        #     logit_list.extend(logits.cpu().numpy())  # 각 배치의 logit들을 리스트에 추가
   
    return pred_list#,logit_list

def main(args):

    hparams = dotdict(vars(args))
    # ------------
    # data
    # ------------
    
    monitor = "val_acc"
    mode = "max"
    save_top_k = 1  # best 하나 저장 + Last 저장


    if hparams.use_kfold:

        hparams.kfold_pl_train_return = True

        skf = StratifiedKFold(n_splits=hparams.n_splits, shuffle=True, random_state=42)

        train_info_df = pd.read_csv(hparams["traindata_info_file"])

        models = []
        
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_info_df, train_info_df["target"])
        ):
            
            # logger
            my_loggers = setup_logger(args.use_wandb, args.sweep_mode, args.output_dir)

            # create model checkpoint callback
            checkpoint_callback = []
            checkpoint_callback.append(
                ModelCheckpoint(
                    dirpath=f"{hparams.output_dir}/fold{fold}",
                    save_last=True,
                    save_top_k=save_top_k,
                    monitor=monitor,
                    mode=mode,
                )
            )
            if hparams.early_stopping > 0:
                early_stop = EarlyStopping(
                    monitor="valid_loss",
                    patience=hparams.early_stopping,
                    verbose=False,
                    mode="min",
                )
                checkpoint_callback.append(early_stop)


            # ------------
            # training
            # ------------
            data_mod = data_module.SketchDataModule(train_idx=train_idx,val_idx=val_idx,**hparams)
            trainer_mod = Sketch_Classifier(**hparams)

            trainer, trainer_mod, data_mod = train(trainer_mod, data_mod,my_loggers,checkpoint_callback,**hparams)

            models.append(trainer_mod)

            # ------------
            # testing
            # ------------

            print("start predict validation dataset")

            pred_list = val_pred(trainer,trainer_mod,data_mod,**hparams)
            
            val_info = data_mod.test_dataset.info_df
            # val_info = val_info.iloc[:100]
            val_info["pred"] = pred_list
            val_info.to_csv(
                os.path.join(args.output_dir + f"/fold{fold}", f"{fold}_validation.csv"),
                index=False,
            )

        print("start predict test dataset")

        data_mod = data_module.SketchDataModule(**hparams)
        test_predictions = np.zeros((len(data_mod.test_info_df), hparams.num_classes))
        test_logits = []

        for model in models:
            model.setup("predict")
            predictions = trainer.predict(model, data_mod)

            pred_list = []
            logit_list = []

            for batch in predictions:
                preds, logits = batch
                pred_list.extend(preds.cpu().numpy())
                logit_list.extend(logits.cpu().numpy())

            test_predictions += F.softmax(torch.tensor(logit_list), dim=1).numpy()
            test_logits.append(logit_list)

        test_predictions /= len(models)

        output_df = pd.DataFrame()
        output_df["ID"] = range(len(data_mod.test_info_df))
        output_df["image_path"] = data_mod.test_info_df["image_path"]
        output_df["target"] = test_predictions.argmax(axis=1)

        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        output_df.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)
    
    else:

        # logger
        my_loggers = setup_logger(args.use_wandb, args.sweep_mode, args.output_dir)

        checkpoint_callback = []
        checkpoint_callback.append(
            ModelCheckpoint(
                dirpath=hparams.output_dir,
                save_last=True,
                save_top_k=save_top_k,
                monitor=monitor,
                mode=mode,
            )
        )

        # ------------
        # training
        # ------------
        data_mod = data_module.SketchDataModule(**hparams)
        trainer_mod = Sketch_Classifier(**hparams)
        trainer,trainer_mod, data_mod = train(trainer_mod, data_mod,my_loggers,checkpoint_callback,**hparams)

        #
        # ------------
        # testing
        # ------------

        best_model_path = checkpoint_callback[0].best_model_path
        best_model = Sketch_Classifier.load_from_checkpoint(best_model_path, **hparams)

        print("start predict")

        # Make predictions
        predictions = trainer.predict(best_model, datamodule=data_mod)
        pred_list = [pred.cpu().numpy() for batch in predictions for pred in batch]

        # Prepare test information DataFrame
        test_info = data_mod.test_dataset.info_df
        test_info["target"] = pred_list
        test_info["ID"] = test_info["image_path"].str.extract(r"(\d+)").astype(int)
        test_info.sort_values(by=["ID"], inplace=True)
        test_info = test_info[["ID", "image_path", "target"]]

        # Save predictions to CSV
        test_info.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)

        print("start predict validation dataset")

        # # Prepare Validation information DataFrame
        val_list = val_pred(trainer,trainer_mod,data_mod,**hparams)
        val_info = data_mod.test_dataset.info_df
        # val_info = val_info.iloc[:100]
        val_info["pred"] = val_list

        # Save Validation predictions to CSV
        val_info.to_csv(os.path.join(args.output_dir, "validation.csv"), index=False)


if __name__ == "__main__":

    pl.seed_everything(42)

    # ------------
    # Load arguments and configuration
    # ------------
    config = load_config("configs/base_config.yaml")
    args = parse_args(config)

    # ------------
    # Set up output directory
    # ------------
    current_time = strftime("%m-%d_0", gmtime())
    pt = "O" if args.pretrained else "X"
    name_str = (
        f"{args.model_name}-{args.batch_size}-{args.learning_rate}"
        + f"-{args.optim}-{pt}-{args.exp_name}"
    )

    # args.output_dir = os.path.join(args.base_output_dir, args.exp_name + "_" + current_time)
    args.output_dir = os.path.join(args.base_output_dir, name_str + "_" + current_time)

    # Check for existing directory and increment if necessary
    if os.path.isdir(args.output_dir):
        while True:
            cur_exp_number = int(args.output_dir[-2:].replace("_", ""))
            args.output_dir = args.output_dir[:-2] + "_{}".format(cur_exp_number + 1)
            if not os.path.isdir(args.output_dir):
                break

    # gpus
    args.gpus = [int(i) for i in str(args.gpus).split(",")]

    if args.sweep_mode:
        run_sweep()
    else:
        main(args)
