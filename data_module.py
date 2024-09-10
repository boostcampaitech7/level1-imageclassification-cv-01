import os
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader
# from torchvision import transforms


from data_sets import folder_dataset, base_dataset
from select_transforms import TransformSelector

class SketchDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_data_dir = kwargs['train_data_dir'] 
        self.test_data_dir = kwargs['test_data_dir'] 
        self.batch_size = kwargs['batch_size']
        self.data_name = kwargs['data_name']


        self.train_data_dir = kwargs['train_data_dir']
        self.test_data_dir = kwargs['test_data_dir']

        self.num_workers = kwargs['num_workers']
        
        transform_selector = TransformSelector(kwargs['transform_name'])
        self.train_info_df = pd.read_csv(kwargs['traindata_info_file'])
        self.test_info_df = pd.read_csv(kwargs['testdata_info_file'])

        self.train_transform = transform_selector.get_transform(True) #is train
        self.test_transform = transform_selector.get_transform(False)

        
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # full_trainset = folder_dataset.CustomImageFolderDataset(self.train_data_dir, transform=self.train_transform)
            
            full_trainset = train_data(self.data_name, self.train_transform, self.train_data_dir,info_df=self.train_info_df) #TODO val,train 분리
            
            self.train_dataset, self.val_dataset = random_split(
                full_trainset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     # self.test_dataset = folder_dataset.CustomImageFolderDataset(self.test_data_dir, transform=self.test_transform)
        #     self.test_dataset = test_data(self.data_name, self.test_transform, self.test_data_dir,info_df=self.test_info_df)
        
        if stage == "predict":
            self.test_dataset = test_data(self.data_name, self.test_transform, self.test_data_dir,info_df=self.test_info_df)



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def train_data(data_name, transforms, train_data_dir='./', info_df = None, is_inference = False):
    if data_name == 'base':
        return base_dataset.CustomDataset(train_data_dir,info_df,transforms,is_inference)
    elif data_name == 'folder':
        return folder_dataset.CustomImageFolderDataset(train_data_dir , transform=transforms)
    else:
        raise ValueError('not a correct train data name', data_name)

#TODO 
# def val_data()


def test_data(data_name, transforms, test_data_dir='./', info_df = None, is_inference = True):
    if data_name == 'base':
        return base_dataset.CustomDataset(test_data_dir,info_df,transforms,is_inference)
    elif data_name == 'folder':
         return folder_dataset.CustomImageFolderDataset(test_data_dir, transform=transforms)
    else:
        raise ValueError('not a correct test data name', data_name)
    
