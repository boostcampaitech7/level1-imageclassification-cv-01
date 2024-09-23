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
from torchvision.datasets.mnist import MNIST

import data_module
from pl_trainer import Sketch_Classifier
from utils.util import dotdict

import wandb

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



def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args(config):
    parser = ArgumentParser()

    # Set defaults from config file, but allow override via command line
    parser.add_argument(
        "--exp_name", type=str, default=config.get("exp_name")
    )  # 현재 실험 이름
    parser.add_argument(
        "--base_output_dir", type=str, default=config.get("base_output_dir")
    )  # 실험 결과 저장 폴더
    parser.add_argument("--gpus", type=str, default=config.get("gpus"))

    parser.add_argument("--batch_size", type=int, default=config.get("batch_size"))
    parser.add_argument("--epochs", type=int, default=config.get("epochs"))
    parser.add_argument(
        "--learning_rate", type=float, default=config.get("learning_rate")
    )

    parser.add_argument('--num_cnn_classes', type=int, default=config.get('num_cnn_classes'))  # CNN 분류 클래스 수

    parser.add_argument(
        "--model_type", type=str, default=config.get("model_type")
    )  # backbone 타입
    parser.add_argument(
        "--model_name", type=str, default=config.get("model_name")
    )  # torchvision, timm을 위한 model 이름
    parser.add_argument("--pretrained", type=bool, default=config.get("pretrained"))
    parser.add_argument(
        "--data_name", type=str, default=config.get("data_name")
    )  # dataset 이름
    parser.add_argument(
        "--transform_name", type=str, default=config.get("transform_name")
    )
    parser.add_argument("--num_classes", type=int, default=config.get("num_classes"))
    parser.add_argument("--optim", type=str, default=config.get("optim"))
    parser.add_argument("--weight_decay", type=str, default=config.get("weight_decay"))
    parser.add_argument("--loss", type=str, default=config.get("loss"))
    parser.add_argument(
        "--cos_sch", type=int, default=config.get("cos_sch")
    )  # cos 주기
    parser.add_argument("--warm_up", type=int, default=config.get("warm_up"))
    parser.add_argument(
        "--early_stopping", type=int, default=config.get("early_stopping")
    )

    parser.add_argument(
        "--train_data_dir", type=str, default=config.get("train_data_dir")
    )
    parser.add_argument(
        "--traindata_info_file", type=str, default=config.get("traindata_info_file")
    )
    parser.add_argument(
        "--test_data_dir", type=str, default=config.get("test_data_dir")
    )
    parser.add_argument(
        "--testdata_info_file", type=str, default=config.get("testdata_info_file")
    )

    parser.add_argument(
        "--use_wandb", type=int, default=config.get("use_wandb")
    )  # wandb 사용?
    parser.add_argument(
        "--num_workers", type=str, default=config.get("num_workers")
    )  # dataloader 옵션 관련
    parser.add_argument("--cutmix_mixup", type=str, default=config.get("cutmix_mixup"))
    parser.add_argument("--cutmix_ratio", type=int, default=config.get("cutmix_ratio"))
    parser.add_argument("--mixup_ratio", type=int, default=config.get("mixup_ratio"))
    parser.add_argument("--kfold_pl_train_return", type=str, default=False)
    parser.add_argument(
        "--mixed_precision", type=bool, default=config.get("mixed_precision")
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=config.get("accumulate_grad_batches"),
    )
    parser.add_argument("--sweep_mode", type=bool, default=config.get("sweep_mode"))

    return parser.parse_args()


def main(args):

    if args.sweep_mode:
        with wandb.init() as run:
            args.learning_rate = run.config.learning_rate
            args.batch_size = run.config.batch_size
            args.optim = run.config.optimizer
            args.loss = run.config.loss
            args.cutmix_ratio = run.config.cutmix_ratio
            args.mixup_ratio = run.config.mixup_ratio
            # loss_type에 따른 추가 파라미터 설정
            if args.loss== "CE":
                args.label_smoothing = run.config.label_smoothing
            else:
                args.label_smoothing = 0  # Label smoothing이 없을 경우 기본값
            
            if args.loss == "Focal":
                args.alpha_val = run.config.focal_alpha
                args.gamma_val = run.config.focal_gamma
            else:
                args.alpha_val = 0
                args.gamma_val = 0
            run.name = f"lr({args.learning_rate:.3e})_bs{args.batch_size}_{args.optim}_{args.loss}_cutmix({args.cutmix_ratio:.3e})_mixup({args.mixup_ratio:.3e})_label_smoothing({args.label_smoothing})_Focal_alpha{args.alpha_val, args.gamma_val}"
            

    hparams = dotdict(vars(args))

    # ------------
    # data
    # ------------
    data_mod = data_module.SketchDataModule(**hparams)

    # logger
    csv_logger = CSVLogger(save_dir=hparams.output_dir, name="result")
    my_loggers = [csv_logger]
    if args.use_wandb:
        wandb_logger = WandbLogger(
            save_dir=hparams.output_dir,
            # name=run.name if run else os.path.basename(hparams.output_dir),
            # project="sketch classification",
        )
        my_loggers.append(wandb_logger)

    checkpoint_dir = os.path.join(hparams.output_dir, run.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # create model checkpoint callback
    monitor = "val_acc"  # validation loss를 모니터링 
    mode = "min"
    save_top_k = 1  # best 하나 저장 + Last 저장
    checkpoint_callback = []
    checkpoint_callback.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_last=True,
            save_top_k=save_top_k,
            monitor=monitor,
            mode=mode,
        )
    )

    if hparams.early_stopping > 0:
        early_stop = EarlyStopping(
            monitor="valid_loss", # val loss가 안돌아 간다...?
            patience=hparams.early_stopping,
            verbose=False,
            mode="min",
        )
        checkpoint_callback.append(early_stop)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        logger=my_loggers,
        accelerator="cpu" if hparams.gpus == 0 else "gpu",
        precision='16-mixed' if hparams.gpus != 0 else 32,
        devices=None if hparams.gpus == 0 else hparams.gpus,
        callbacks=checkpoint_callback,
        max_epochs=hparams.epochs,
        accumulate_grad_batches=(
            1
            if hparams.accumulate_grad_batches <= 0
            else hparams.accumulate_grad_batches
        ),
    )

    trainer_mod = Sketch_Classifier(**hparams)
    trainer.fit(trainer_mod, data_mod)

    # ------------
    # testing
    # ------------
    best_model_path = checkpoint_callback[0].best_model_path
    best_model = Sketch_Classifier.load_from_checkpoint(best_model_path, **hparams)

    print("start predict")
    predictions = trainer.predict(best_model, datamodule=data_mod)
    pred_list = [pred.cpu().numpy() for batch in predictions for pred in batch]

    test_info = data_mod.test_dataset.info_df
    test_info["target"] = pred_list

    test_info["ID"] = test_info["image_path"].str.extract(r"(\d+)").astype(int)
    test_info.sort_values(by=["ID"], inplace=True)
    test_info = test_info[["ID", "image_path", "target"]]
    
    test_info.to_csv(os.path.join(checkpoint_dir, "output.csv"), index=False)

    print("start predict validation dataset")
    
    data_mod.test_data_dir = data_mod.train_data_dir
    data_mod.test_info_df = data_mod.val_info_df
    data_mod.setup(stage="predict")

    validations = trainer.predict(best_model, dataloaders=data_mod.predict_dataloader())
    val_list = [val.cpu().numpy() for batch in validations for val in batch]
    val_info = data_mod.test_dataset.info_df
    val_info["pred"] = val_list
    val_info.to_csv(os.path.join(checkpoint_dir, "validation.csv"), index=False)

if __name__ == "__main__":

    pl.seed_everything(42)

    # ------------
    # args
    # ------------

    config = load_config("config.yaml")
    args = parse_args(config)

    ## output_dir
    current_time = strftime("%m-%d_0", gmtime())
    pt = "O" if args.pretrained else "X"
    name_str = (
        f"{args.model_name}-{args.batch_size}-{args.learning_rate}"
        + f"-{args.optim}-{pt}-{args.exp_name}"
    )
    # args.output_dir = os.path.join(args.base_output_dir, args.exp_name + "_" + current_time)
    args.output_dir = os.path.join(args.base_output_dir, name_str + "_" + current_time)
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
