# BoostCamp AI Tech 7th CV-01 Project

<p align="center">
  <img src="https://github.com/user-attachments/assets/301a97b9-2caf-4ae2-a895-17ee0b1a5711" alt="image" width="300"/>
</p>

## Project Overview
This project is part of BoostCamp AI Tech and focuses on developing a classfication model using sketch data.

## Table of Contents
- [Installation](#installation)
  - [VSCode SSH Connection](#vscode-ssh-connection)
  - [Install Packages](#install-packages)
- [Tree Structure](#tree-structure)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Train & Test](#train--test)
  - [Additional Setting](#additional-setting)

## Installation

<details>
  <summary id="vscode-ssh-connection">VSCode SSH Connection</summary>

#### 1. Run OpenVPN 

#### 2. Add New SSH Host

  - Press `Ctrl+Shift+P` on VSCode to open the command palette
  - Select `Remote-SSH: Add New SSH Host`.  

  - Enter the information like below:  
      ```bash
      ssh -p 31678 root@10.28.224.95
      ```
    
  - Add Identity Line (path to your SSH config file) in configuration file
      ```bash
      Host {IP}
      HostName {IP}
      Port {Port}
      User root
      IdentityFile C:\Users\HOME\Downloads\{pem파일}.pem
      ```

#### 3. RUN New SSH

  - Click the right arrow button on the SSH setting tab to connect.

</details>

<details>
  <summary id="install-packages">Install Packages</summary>
    
  - After connecting to the server, follow these steps to install the required packages:
      ```bash
      # Update and install necessary packages
      apt-get update -y && apt-get upgrade -y && \
      apt-get install -y libgl1-mesa-glx libglib2.0-0 wget git curl tmux sudo
      
      # Clone the project repository
      git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-01.git
      cd level1-imageclassification-cv-01

      # Prepare the data
      tar -zxvf data.tar.gz && rm data.tar.gz

      # Install Python dependencies
      pip install -r requirements.txt
      ```

</details>

## Run

```
python tools/train_and_predict.py
```

## Tree Structure

```
level1-imageclassification-cv-01/|-- README.md
|
|─── baseline_codes
|   |─── base_dataset.py
|   |─── baseline_code.ipynb
|   └─── eda.ipynb
|
|─── configs
|   └─── base_config.yaml
|
|─── docker
|   └─── Dockerfile
|
|─── notebooks
|   |─── augmentation_visualization.ipynb
|   |─── data_similarity_remove.ipynb
|   |─── eda.ipynb
|   |─── grad_cam.ipynb
|   |─── model_structure_confirm.ipynb
|   |─── pre_test_score.ipynb
|   └─── validation_data_check.ipynb
|
|─── requirements.txt
|
|─── src
|   |
|   |─── __init__.py
|   |─── data
|   |   |─── __init__.py
|   |   |─── base_dataset.py
|   |   |─── data_module.py
|   |   |─── folder_dataset.py
|   |   |─── swin_custom_dataset.py
|   |   └─── transforms.py
|   |
|   |─── models
|   |   |─── __init__.py
|   |   |─── base_backbone.py
|   |   |─── clip_backbone.py
|   |   |─── cnnvit_backbone.py
|   |   |─── conv_backbone.py
|   |   |─── ensemble.py
|   |   |─── model_selector.py
|   |   └─── swin_backbone.py
|   |
|   |─── training
|   |   |─── __init__.py
|   |   |
|   |   |─── losses
|   |   |   |─── __init__.py
|   |   |   |─── ce_loss.py
|   |   |   |─── focal_loss.py
|   |   |   |─── loss_selector.py
|   |   |   └─── swin_combined_loss.py
|   |   |
|   |   └─── trainer.py
|   |
|   └─── utils
|       |─── __init__.py
|       └─── util.py
|
|─── streamlit
|   |─── README.md
|   |─── data_analize_page.py
|   └─── requirements.txt
|
└─── tools
    |─── __init__.py
    └─── train_and_predict.py
```

## Usage
<details> 
  <summary id="data-preprocessing">Data Preprocessing</summary> 

- #### Data Augmentation using OpenCV

  **Augmentation methods that cannot be handled by `[Torchvision.transforms](https://pytorch.org/vision/0.9/transforms.html)' or '[Albumentations.Transforms](https://albumentations.ai/docs/getting_started/transforms_and_targets/)' are performed using OpenCV. The augmented data is then added to the data folder.** 

- #### Data Augmentation using Transform  
  **Modify the `TransformSelector` class in `src/data/transforms.py` as follows:** 
    
    ```python
    class TransformSelector:
        """
        Class for selecting the image transformation library.
        """
        def __init__(self, transform_type: str):
            # Ensure the transformation library is supported
            if transform_type in ["torchvision", "albumentations", "aug_test"]:
                self.transform_type = transform_type
            else:
                raise ValueError("Unknown transformation library specified.")

        def get_transform(self, is_train: bool):
            # Return the appropriate transform object based on the library
            if self.transform_type == 'torchvision':
                transform = TorchvisionTransform(is_train=is_train)
            elif self.transform_type == 'albumentations':
                transform = AlbumentationsTransform(is_train=is_train)
            elif self.transform_type == 'aug_test':
                transform = A_aug_test(is_train=is_train)
            
            return transform
    ```

</details> 
<details>
  <summary id="model-architecture">Model Architecture</summary> 
    
  - You can use pre-built models from the `timm` library or `torchvision`. To customize, you can create new models under the `src/models/` folder and modify them as needed. 
  </details> 


<details> 
  <summary id="additional-setting">Additional Setting</summary> 

  - Modify `configs/base_config.yaml` to adjust various training and model parameters: 

      ```yaml
      ######################
      # 실험 설정
      ######################
      use_wandb: True
      exp_name: resnext50_32x4d
      gpus: 0
      
      ######################
      # 모델 설정
      ######################
      model_type: timm
      model_name: resnext50_32x4d
      pretrained: True
      
      ######################
      # 데이터 설정
      ######################
      train_data_dir: ./data/train
      test_data_dir: ./data/test
      base_output_dir: ./result
      num_classes: 500
      data_name: base
      traindata_info_file: ./data/train.csv
      testdata_info_file: ./data/test.csv
      
      ######################
      # 학습 설정
      ######################
      epochs: 100
      learning_rate: 0.0001
      num_workers: 8
      cos_sch: 50
      early_stopping: 10
      warm_up: 10
      batch_size: 64
      weight_decay: 0.0
      loss: CE
      transform_name: albumentations
      optim: AdamW
      mixed_precision: True
      num_cnn_classes: 20
      
      ######################
      # 데이터 증강 설정
      ######################
      cutmix_mixup: mixup_cutmix
      cutmix_ratio: 0.2
      mixup_ratio: 0.2
      
      ######################
      # 교차 검증 및 기타 설정
      ######################
      n_splits: 5
      accumulate_grad_batches: 8
      sweep_mode: False
      use_kfold: True
      ```
      
</details>
