import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
from sklearn.model_selection import train_test_split

from src.data.folder_dataset import CustomImageFolderDataset
from src.data.base_dataset import CustomDataset
from src.data.transforms import TransformSelector
from src.data.swin_custom_dataset import SwinCustomDataset


class SketchDataModule(pl.LightningDataModule):

    def __init__(self, train_idx = None, val_idx= None, **kwargs):
        super().__init__()

        self.train_data_dir = kwargs["train_data_dir"]
        self.test_data_dir = kwargs["test_data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.data_name = kwargs["data_name"]

        # self.train_data_dir = kwargs['train_data_dir']
        # self.test_data_dir = kwargs['test_data_dir']

        self.num_workers = kwargs["num_workers"]

        transform_selector = TransformSelector(kwargs["transform_name"])
        self.train_info_df = pd.read_csv(kwargs["traindata_info_file"])
        self.test_info_df = pd.read_csv(kwargs["testdata_info_file"])
        

        if kwargs['use_kfold'] and val_idx is not None:
            self.train_info_df, self.val_info_df = self.train_info_df.iloc[train_idx], self.train_info_df.iloc[val_idx]
            
        else: 
            ## 특정 class sample이 1일 때 해결 
            class_counts = self.train_info_df["target"].value_counts()
            low_sample_classes = class_counts[class_counts < 2].index
            low_sample_df = self.train_info_df[self.train_info_df["target"].isin(low_sample_classes)]
            remaining_df = self.train_info_df[~self.train_info_df["target"].isin(low_sample_classes)]

            self.train_info_df, self.val_info_df = train_test_split(
                remaining_df,
                test_size=0.2,  # validation 비율 (20%)
                random_state=42,  # 랜덤 시드 고정
                stratify=remaining_df["target"],  # 라벨을 기준으로 비율 유지
            )
            self.train_info_df = pd.concat([self.train_info_df, low_sample_df])



        self.train_transform = transform_selector.get_transform(True)  # is train
        self.test_transform = transform_selector.get_transform(False)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            # full_trainset = train_data(self.data_name, self.train_transform, self.train_data_dir,info_df=self.train_info_df) #TODO val,train 분리

            # self.train_dataset, self.val_dataset = random_split(
            #     full_trainset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            # )
            self.train_dataset = train_data(
                self.data_name,
                self.train_transform,
                self.train_data_dir,
                info_df=self.train_info_df,
            )
            self.val_dataset = val_data(
                self.data_name,
                self.test_transform,
                self.train_data_dir,
                info_df=self.val_info_df,
            )

        if stage == "predict":
            self.test_dataset = test_data(
                self.data_name,
                self.test_transform,
                self.test_data_dir,
                info_df=self.test_info_df,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def train_data(
    data_name, transforms, train_data_dir="./", info_df=None, is_inference=False
):
    if data_name == "base":
        return CustomDataset(
            train_data_dir, info_df, transforms, is_inference
        )
    elif data_name == "folder":
        return CustomImageFolderDataset(
            train_data_dir, transform=transforms
        )
    elif data_name == 'swin_data':
        return SwinCustomDataset(
            train_data_dir, info_df, transforms, is_inference
        )
    else:
        raise ValueError("not a correct train data name", data_name)


# TODO
# def val_data()
def val_data(
    data_name, transforms, train_data_dir="./", info_df=None, is_inference=False
):
    if data_name == "base":
        return CustomDataset(
            train_data_dir, info_df, transforms, is_inference
        )
    elif data_name == "folder":
        return CustomImageFolderDataset(
            train_data_dir, transform=transforms
        )
    elif data_name == 'swin_data':
        return SwinCustomDataset(
            train_data_dir, info_df, transforms, is_inference
        )
    else:
        raise ValueError("not a correct train data name", data_name)


def test_data(
    data_name, transforms, test_data_dir="./", info_df=None, is_inference=True
):
    if data_name == "base":
        return CustomDataset(
            test_data_dir, info_df, transforms, is_inference
        )
    elif data_name == "folder":
        return CustomImageFolderDataset(
            test_data_dir, transform=transforms
        )
    elif data_name == 'swin_data':
        return SwinCustomDataset(
            test_data_dir, info_df, transforms, is_inference
        )
    else:
        raise ValueError("not a correct test data name", data_name)
