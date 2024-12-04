from setuptools import setup, find_packages

setup(
    name="diffusion-models",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.2',
        'Pillow>=8.3.1',
        'tqdm>=4.62.2',
        'wandb>=0.12.0',
        'ruamel.yaml>=0.17.21',
        'torchmetrics>=0.11.4',
    ],
) 