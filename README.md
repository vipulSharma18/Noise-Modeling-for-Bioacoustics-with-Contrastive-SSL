>Note: This work was originally done as part of CSCI2980 Reading and Research coursework at Brown University during my Master's in Computer Science.

# SSL-Bioacoustics

Create virtual env and install from source as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Force update if stable-ssl fork had to be modified.

```bash
pip install --force-reinstall --no-deps "stable-ssl @ git+https://github.com/vipulSharma18/stable-SSL.git@main#egg=stable-ssl"
```

## Experiment commands:
### CBC2020:
Supervised non-sequential spectrogram evaluation:
```bash
python run.py -m mode=one_gpu_slurm experiment=cbc2020/static_supervised_cbc2020 user=vsharm44 ++trainer.hardware.seed=0,10,20 ++module.backbone.name=resnet18,resnet50
```

Supervised sequential/sliding window spectrogram evaluation:
```bash
python run.py -m mode=one_gpu_slurm experiment=cbc2020/sequential_supervised_cbc2020 user=vsharm44 ++trainer.hardware.seed=0,10,20 ++module.backbone.name=cnn_lstm
```

SSL sequential/sliding window spectrogram evaluation (CPC):
```bash
python run.py -m mode=one_gpu_slurm experiment=cbc2020/sequential_cpc_cbc2020 user=vsharm44 ++trainer.hardware.seed=0,10,20 ++module.backbone.name=cnn_lstm
```

SSL sequential/sliding window spectrogram evaluation (CPC and random noise):
```bash
python run.py -m mode=one_gpu_slurm experiment=cbc2020/sequential_cpc_noise_cbc2020 user=vsharm44 ++trainer.hardware.seed=0,10,20 ++module.backbone.name=cnn_lstm ++trainer.data.train.dataset.noise_transform.snr=0.1,10
```

### UrbanSound8k:
Reproduce SimCLR Urban Sound 8k results:
Pretrained:
```bash
python run.py -m mode=one_gpu_slurm experiment=urbansound8k/simclr_urbansound8k user=vsharm44 ++trainer.data.train.dataset.fold=1,2,3,4,5,6,7,8,9,10 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=True ++logger.wandb.group=simclr_urbansound8k_pretrainedresnet50
```
Untrained:
```bash
python run.py -m mode=one_gpu_slurm experiment=urbansound8k/simclr_urbansound8k user=vsharm44 ++trainer.data.train.dataset.fold=1,2,3,4,5,6,7,8,9,10 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=False ++logger.wandb.group=simclr_urbansound8k_untrainedresnet50
```

Reproduce Supervised Urban Sound 8k results:
Pretrained:
```bash
python run.py -m mode=one_gpu_slurm experiment=urbansound8k/supervised_urbansound8k user=vsharm44 ++trainer.data.train.dataset.fold=1,2,3,4,5,6,7,8,9,10 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=True ++logger.wandb.group=supervised_urbansound8k_pretrainedresnet50
```
Untrained:
```bash
python run.py -m mode=one_gpu_slurm experiment=urbansound8k/supervised_urbansound8k user=vsharm44 ++trainer.data.train.dataset.fold=1,2,3,4,5,6,7,8,9,10 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=False ++logger.wandb.group=supervised_urbansound8k_untrainedresnet50
```

### Birdsong:
Shuffle and learn Birdsong:
Untrained:
```bash
python run.py -m mode=one_gpu_slurm experiment=birdsong/shuffle_and_learn_birdsong user=vsharm44 ++trainer.hardware.seed=0 ++module.backbone.weights=False ++logger.wandb.group=shuffle_and_learn_birdsong_test_untrainedalexnet
python run.py -m mode=one_gpu_slurm experiment=birdsong/shuffle_and_learn_birdsong user=vsharm44 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=False ++logger.wandb.group=shufflelearn_birdsong_untrainedalexnet
```
Pretrained on ImageNet:
```bash
python run.py -m mode=one_gpu_slurm experiment=birdsong/shuffle_and_learn_birdsong user=vsharm44 ++trainer.hardware.seed=0 ++module.backbone.weights=True ++logger.wandb.group=shuffle_and_learn_birdsong_test_pretrainedalexnet
python run.py -m mode=one_gpu_slurm experiment=birdsong/shuffle_and_learn_birdsong user=vsharm44 ++trainer.hardware.seed=0,100,200,300,400 ++module.backbone.weights=True ++logger.wandb.group=shufflelearn_birdsong_pretrainedalexnet
```
