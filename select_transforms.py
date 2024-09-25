import random

import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """

    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations", "aug_test"]:
            self.transform_type = transform_type

        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):

        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        # 또는 추후 구현
        if self.transform_type == "torchvision":
            transform = TorchvisionTransform(is_train=is_train)

        elif self.transform_type == "albumentations":
            transform = AlbumentationsTransform(is_train=is_train)

        elif self.transform_type == "aug_test":
            transform = A_aug_test(is_train=is_train)

        return transform


class TorchvisionTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((224, 224)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(
                        p=0.3
                    ),  # 50% 확률로 이미지를 수직 뒤집기
                    transforms.RandomHorizontalFlip(
                        p=0.3
                    ),  # 50% 확률로 이미지를 수평 뒤집기
                    transforms.RandomRotation(
                        (-15, 15)
                    ),  # 이미지를 -15도에서 15도 사이로 회전
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2
                    ),  # 밝기 및 대비 조정
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환

        transformed = self.transform(image)  # 설정된 변환을 적용

        return transformed  # 변환된 이미지 반환


class AlbumentationsTransform:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
            ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed["image"]  # 변환된 이미지의 텐서를 반환


class A_aug_test:
    def __init__(self, is_train: bool = True):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Resize(224, 224),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
            ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        self.is_train = is_train

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.ElasticTransform(
                        alpha=50, sigma=10, border_mode=cv2.BORDER_REFLECT_101
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=(-0.3, 0.3),
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.Affine(
                        scale=(0.8, 1.2),
                        translate_percent=(0.1, 0.1),
                        rotate=(-30, 30),
                        shear=(-10, 10),
                        mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    # A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def square_padding(self, image):
        h, w, _ = image.shape
        target_size = max(h, w)
        transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=target_size,
                    min_width=target_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            ]
        )
        return transform(image=image)["image"]

    def random_square_padding(self, image):
        h, w, _ = image.shape

        # 이미지가 가로가 더 긴 경우 (세로에 패딩 추가)
        if h < w:
            diff = w - h
            top = random.randint(0, diff)
            bottom = diff - top
            left, right = 0, 0

        # 이미지가 세로가 더 긴 경우 (가로에 패딩 추가)
        elif w < h:
            diff = h - w
            left = random.randint(0, diff)
            right = diff - left
            top, bottom = 0, 0

        # 이미 정사각형인 경우
        else:
            top, bottom, left, right = 0, 0, 0, 0

        # Pad the image to make it square
        pad_transform = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=h + top + bottom,
                    min_width=w + left + right,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
            ]
        )

        return pad_transform(image=image)["image"]

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        if self.is_train:
            image = self.random_square_padding(image)
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed["image"]  # 변환된 이미지의 텐서를 반환
