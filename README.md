부스트 캠프 AI Tech project


## Conda  Start

프로젝트 위치에 data 압축 풀기 
위치 확인 필요

아래 코드 순서대로 
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
select_transforms.py의 TransformSelector 클래스의
get_transform 함수에서 elif 문으로 transform_type을 추가
TransformSelector의 init에 option 확인 추가 

```
main.py (`data_mod = data_module.SketchDataModule(**hparams)`) 
-> data_module.py (`transform_selector = TransformSelector(kwargs['transform_name'])`)
-> select_transforms.py (`TransformSelector init 참고`)
-> select_transforms.py (`get_transform 참고`)
```

## datasets 
data_module.py에서 train_data, test_data 함수에 elif문으로 data_name 추가
생성한 dataset class 반환 
config.yaml에서 data_name 수정 

```
main.py (`data_mod = data_module.SketchDataModule(**hparams)`) 
-> data_module.py (`def train_data`, `def test_data`) 각각 train, test dataset 정의 
```

### backbone 수정 
net.py의 ModelSelector class의 init에 elif 문으로 model_type 추가 
생성한 모델 backbone 반환 
config.yaml에서 model_type 수정


### training 과정 수정?
pl_trainer 복사해서 새 파일 만들기...?



### Optimizer 
pl_trainer.py의 configure_optimizers에서 elif 추가 
config에 optim 설정 변경 


### Loss
losses.py에 모듈 추가, get_loss 함수에 elif문으로 loss_name 추가 후 return
config에 loss 변경


### 각 모듈에 input을 달리하고 싶다 
config에 추가 후 main의 parse_args 함수에 추가 
kwarg인자 값에서 받아 올 수 있다.
main에서는 hparams.키워드 로 불러올 수 있음 



### 데이터가 너무 많아 빨리 테스트 해볼 수 없다?
기본 설정인 base_dataset.py에서 22번째 줄 주석 해제 
 

## 만약 gpu가 2개 이상이다...? 
예시 config gpu를 2,3 (2번째, 3번째 gpu)으로 수정

## 내가 이 Readme를 꾸미겠다 
부탁드립니다

### TODO
- [ ] Resuming from checkpoints
- [X] Loss 선택지 추가
- [X] Optim 선택지 추가 
