부스트 캠프 AI Tech project


## Conda  Start

프로젝트 위치에 data 압축 풀기 
위치확인 필요

```

conda create -n 가상환경이름 python=3.12

conda activate 가상환경이름

conda install --yes --file requirements.txt -c conda-forge

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

```

## Project Tree
TODO 

## Usage
Try `python train.py` to run code.



## 전처리 수정
### Transform
select_transforms.py의 get_transform 함수에서 elif 문으로 transform_type을 추가
생성한 transform class를 반환
config.yaml에서 transform_name 수정 

main.py (`data_mod = data_module.SketchDataModule(**hparams)`) 
-> data_module.py (`transform_selector = TransformSelector(kwargs['transform_name'])`)
-> select_transforms.py (`TransformSelector init 참고`)
-> select_transforms.py (`get_transform 참고`)

## datasets 
data_module.py에서 train_data, test_data 함수에 elif문으로 data_name 추가
생성한 dataset class 반환 
config.yaml에서 data_name 수정 

main.py (`data_mod = data_module.SketchDataModule(**hparams)`) 
-> data_module.py (`def train_data`, `def test_data`) 각각 train, test dataset 정의 


### backbone 수정 
net.py의 ModelSelector class의 init에 elif 문으로 model_type 추가 
생성한 모델 backbone 반환 
config.yaml에서 model_type 수정


### training 과정 수정?
pl_trainer 복사해서 새 파일 만들기...?


### Optimizer 
수정 예정 


### Loss
수정 예정 



### TODO
- [ ] Resuming from checkpoints
- [ ] Loss 선택지 추가
- [ ] Optim 선택지 추가 