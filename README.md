부스트 캠프 AI Tech project


## Conda  Start
`conda create -n 가상환경이름 python=3.12`
`conda activate 가상환경이름`
`conda install --yes --file requirements.txt -c conda-forge`
`conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

## Usage
Try `python train.py -c config.json` to run code.


### TODO
- [ ] Resuming from checkpoints