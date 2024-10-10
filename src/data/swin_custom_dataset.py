import os
from typing import Callable, Union, Tuple

import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class SwinCustomDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        info_df: pd.DataFrame,
        transform: Callable,
        is_inference: bool = False
    ):
        self.root_dir = root_dir  
        self.transform = transform  
        self.is_inference = is_inference
        self.info_df = info_df
        # self.info_df = info_df.iloc[:100]  # TODO
        self.image_paths = self.info_df["image_path"].tolist()
        
        if not self.is_inference:
            self.original_labels = self.info_df['target'].tolist()
            self.large_labels = self.info_df['large_label'].tolist()  # 대분류 라벨
            self.small_labels = self.info_df['small_label'].tolist()  # 중분류 라벨

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(
            self.root_dir, self.image_paths[index]
        )
        image = cv2.imread(
            img_path, cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        if self.transform:
            image = self.transform(image)

        if self.is_inference:
            return image
        else:
            original_label = self.original_labels[index]
            large_label = self.large_labels[index]
            small_label = self.small_labels[index]
            return image, original_label, large_label, small_label
