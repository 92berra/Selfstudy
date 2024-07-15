import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

if not os.path.exists('GANs-Tutorial/Lab1-VanillaGAN/data/target/1/png'):
    os.makedirs('GANs-Tutorial/Lab1-VanillaGAN/data/target/1/png/train')
    os.makedirs('GANs-Tutorial/Lab1-VanillaGAN/data/target/1/png/test')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='GANs-Tutorial/Lab1-VanillaGAN/data/target/1',
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='GANs-Tutorial/Lab1-VanillaGAN/data/target/1',
    train=False,
    download=False,
    transform=transform
)

def save_images(dataset, dataset_type):
    for index, (image, label) in enumerate(dataset):
        image = transforms.ToPILImage()(image)
        file_name = f'GANs-Tutorial/Lab1-VanillaGAN/data/target/1/png/{dataset_type}/{index}_{label}.png'
        image.save(file_name)

save_images(train_dataset, 'train')
save_images(test_dataset, 'test')

print("Finished!")
