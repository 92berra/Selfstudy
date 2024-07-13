# GANs Tutorial (using mps) / Alpha version

For this tutorial, you can train using Macbook GPU(mps).

<br/>

## Environment
- M3 Macbook Pro 
- macOS Sonoma 14.5
- VSCode 1.90.2 (Universal)

<br/>

## Install Python and PyTorch

- <a href='https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html'>Miniconda</a>
- Python 3.9
- PyTorch 2.3.1

```
conda create pytorch-mps python=3.9
conda activate pytorch-mps
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda env update --file requirements.yml
python -m ipykernel install --user --name pytorch-mps --display-name "Python 3.9(pytorch-mps)"
```

<br/>
<br/>
<br/>
<br/>
<br/>

<div align='center'>
    Copyright. 92berra 2024
</div>