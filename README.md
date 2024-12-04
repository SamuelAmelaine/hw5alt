# Diffusion Models Implementation

## Setup

1. Clone the repository:
```bash
git clone https://github.com/SamuelAmelaine/hw5alt.git
cd hw5alt
```

2. Create necessary directories:
```bash
mkdir -p Data/imagenet100_128x128
mkdir -p pretrained
mkdir -p experiments
```

3. Download the data:
- Download imagenet100_128x128.tar.gz from: https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z
- Extract it to the Data folder:
```bash
tar -xvf imagenet100_128x128.tar.gz -C Data/
```

4. Download pretrained VAE weights:
- Download model.ckpt and place it in the pretrained folder

5. Install requirements:
```bash
pip install torch torchvision torchaudio
pip install tqdm pillow wandb ruamel.yaml torchmetrics
```

## Training

Basic DDPM training:
```bash
python train.py --config configs/ddpm.yaml
```

With DDIM sampling:
```bash
python train.py --config configs/ddpm.yaml --use_ddim True
```

With Latent DDPM:
```bash
python train.py --config configs/ddpm.yaml --latent_ddpm True
```

With Classifier-Free Guidance:
```bash
python train.py --config configs/ddpm.yaml --use_cfg True
```

## Inference & Evaluation

```bash
python inference.py --ckpt path/to/checkpoint.pth
```

## Project Structure
```
.
├── configs/
│   └── ddpm.yaml          # Configuration file
├── models/                # Model implementations
├── pipelines/             # Generation pipelines
├── schedulers/            # DDPM/DDIM schedulers
├── utils/                 # Utility functions
├── train.py              # Training script
└── inference.py          # Inference & evaluation script
```

## Features
- Basic DDPM implementation
- DDIM sampling support
- Latent space diffusion with VAE
- Classifier-Free Guidance
- FID and IS evaluation metrics
