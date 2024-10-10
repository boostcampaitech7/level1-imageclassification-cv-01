import os

import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2


# https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
# https://lightning.ai/docs/pytorch/stable/advanced/speed.html

class CustomImageFolderDataset(datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       )
        self.root = root
        self.transform = transform
        self.target_transform = transform



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = Image.fromarray(np.asarray(sample)[:,:,::-1])
    
        
        
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample, target