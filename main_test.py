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

# TODO
# test option 따로 만들기
# model ckpt 파일 불러오기
# pin mem등 pl에서 정하는 가속 설정
# https://lightning.ai/docs/pytorch/stable/advanced/speed.html
# wandb team space 생성
# data module 분리 -> validataion data 분리, 전처리가 상이할 수 있음


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
    parser.add_argument(
        "--model_type", type=str, default=config.get("model_type")
    )  # backbone 타입
    parser.add_argument(
        "--model_name", type=str, default=config.get("model_name")
    )  # torchvision, timm을 위한 model 이름
    parser.add_argument(
        "--data_name", type=str, default=config.get("data_name")
    )  # dataset 이름
    parser.add_argument(
        "--transform_name", type=str, default=config.get("transform_name")
    )
    parser.add_argument("--num_classes", type=int, default=config.get("num_classes"))

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
        "--num_workers", type=str, default=config.get("num_workers")
    )  # dataloader 옵션 관련
    parser.add_argument(
        "--mixed_precision", type=bool, default=config.get("mixed_precision")
    )
    parser.add_argument(
        "--ckpt", type=str, default=config.get("ckpt")
    )


    return parser.parse_args()


def main(args):

    hparams = dotdict(vars(args))
    # ------------
    # data
    # ------------
    data_mod = data_module.SketchDataModule(**hparams)

    # logger
    csv_logger = CSVLogger(save_dir=hparams.output_dir, name="result")
    my_loggers = [csv_logger]


    trainer = pl.Trainer(
        logger=my_loggers,
        accelerator="cpu" if hparams.gpus == 0 else "gpu",
        precision='16-mixed' if hparams.gpus != 0 else 32,  # CPU에서는 32-bit precision
        devices=None if hparams.gpus == 0 else hparams.gpus,
        max_epochs=hparams.epochs,
    )


    # ------------
    # testing
    # ------------

    best_model_path = hparams.ckpt

    best_model = Sketch_Classifier.load_from_checkpoint(best_model_path, **hparams)

    print("start predict")

    predictions = trainer.predict(best_model, datamodule=data_mod)
    pred_list = [pred.cpu().numpy() for batch in predictions for pred in batch]

    test_info = data_mod.test_dataset.info_df
    test_info["target"] = pred_list

    test_info["ID"] = test_info["image_path"].str.extract(r"(\d+)").astype(int)
    test_info.sort_values(by=["ID"], inplace=True)
    test_info = test_info[["ID", "image_path", "target"]]

    test_info.to_csv(os.path.join(args.output_dir, "output.csv"), index=False)

    print("start predict validation dataset")

    data_mod.test_data_dir = data_mod.train_data_dir
    data_mod.test_info_df = data_mod.val_info_df
    data_mod.setup(stage="predict")

    validations = trainer.predict(best_model, dataloaders=data_mod.predict_dataloader())
    val_list = [val.cpu().numpy() for batch in validations for val in batch]
    val_info = data_mod.test_dataset.info_df
    val_info["pred"] = val_list
    val_info.to_csv(os.path.join(args.output_dir, "validation.csv"), index=False)


if __name__ == "__main__":

    pl.seed_everything(42)

    # ------------
    # args
    # ------------

    config = load_config("config.yaml")
    args = parse_args(config)

    ## output_dir
    current_time = strftime("%m-%d_0", gmtime())
    name_str = (
        f"{args.model_name}-{args.batch_size}-{args.learning_rate}"
        + f"-{args.exp_name}"
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

    main(args)
