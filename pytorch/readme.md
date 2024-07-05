# PyTorch

## Contents
[VanillaGAN with FashionMNIST] (##1.-Vanilla-GAN-with-FashionMNIST)

<br/>
<br/>
<br/>

## 1. Vanilla GAN with FashionMNIST
### Environment
- M3 Macbook Pro 
- macOS Sonoma 14.5
- VSCode 1.90.2 (Universal)

<br/>

### Install Python and PyTorch

- <a href='https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html'>Miniconda</a>
- Python 3.9
- PyTorch 2.3.1

```
conda create pytorch-mps python=3.9
conda activate pytorch-mps
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda env update --file requirements.uml
python -m ipykernel install --user --name pytorch-mps --display-name "Python 3.9(pytorch-mps)"
```

<br/>

### How to run 

If you don't download the FashionMNIST datasets, edit this code. download=True.

```
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    #download=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

<br/>

```
python ex06-Vanilla-GAN-fashionMNIST.py
```

<br/>

Create gif using generated images.

```
python ex06-create-gif.py
```

<br/>

### Result

<img src='result/sample_animation.gif'/>

<br/>
<br/>
<br/>
<br/>
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>