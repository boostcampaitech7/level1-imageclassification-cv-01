# BoostCamp AI Tech 7th CV-01 Project

<p align="center">
  <img src="https://github.com/user-attachments/assets/06679804-27c5-49ae-851a-7b06f552ae47" alt="image" width="300"/>
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

## Tree Structure

```
level1-imageclassification-cv-01/
|   .flake8
|   .gitignore
|   baseline_code.ipynb
|   config.yaml
|   data_module.py
|   eda.ipynb
|   losses.py
|   main.py
|   net.py
|   pl_trainer.py
|   README.md
|   requirements.txt
|   select_transforms.py
|   test.ipynb
|
├─── .github
|       .keep
|
|─── backbone
|       base_backbone.py
|       conv_backbone.py
|       __init__.py
|
|─── data_sets
|       base_dataset.py
|       folder_dataset.py
|
|─── streamlit
|       data_analize_page.py
|       README.md
|       requirements.txt
|
|─── utils
     |── util.py
     |── __init__.py
```

## Usage
<details> 
  <summary id="data-preprocessing">Data Preprocessing</summary> 

- #### Data Augmentation using OpenCV

  **Augmentation methods that cannot be handled by `[Torchvision.transforms](https://pytorch.org/vision/0.9/transforms.html)' or '[Albumentations.Transforms](https://albumentations.ai/docs/getting_started/transforms_and_targets/)' are performed using OpenCV. The augmented data is then added to the data folder.** 

- #### Data Augmentation using Transform  
  **Modify the `TransformSelector` class in `select_transforms.py` as follows:** 
    
    ```python
    class TransformSelector:
        """
        Class for selecting the image transformation library.
        """
        def __init__(self, transform_type: str):
            # Ensure the transformation library is supported
            if transform_type in ["torchvision", "albumentations"]:
                self.transform_type = transform_type
            else:
                raise ValueError("Unknown transformation library specified.")

        def get_transform(self, is_train: bool):
            # Return the appropriate transform object based on the library
            if self.transform_type == 'torchvision':
                transform = TorchvisionTransform(is_train=is_train)
            elif self.transform_type == 'albumentations':
                transform = AlbumentationsTransform(is_train=is_train)
            
            return transform
    ```

</details> 
<details>
  <summary id="model-architecture">Model Architecture</summary> 
    
  - You can use pre-built models from the `timm` library or `torchvision`. To customize, you can create new models under the `backbone` folder and modify them as needed. 
  </details> 

<details> 
  <summary id="train--test">Train & Test</summary> 

  - To train and test the model, simply run the following command: 
      ```bash 
      python main.py 
      ``` 

</details> 

<details> 
  <summary id="additional-setting">Additional Setting</summary> 

  - Modify `config.yaml` to adjust various training and model parameters: 

      ```yaml
      exp_name: test
      batch_size: 128
      epochs: 1
      learning_rate: 0.01
      gpus: 0
      model_type: timm
      # for torchvision and timm
      model_name: resnet18
      pretrained: False
      train_data_dir: ./data/train
      test_data_dir: ./data/test
      base_output_dir: ./result
      num_classes: 500
      use_wandb: True
      data_name: base
      num_workers: 1
      optim: Adam
      loss: CE
      # select_transforms.py
      transform_name: torchvision
      traindata_info_file: ./data/train.csv
      testdata_info_file: ./data/test.csv
      ```
      
</details>
