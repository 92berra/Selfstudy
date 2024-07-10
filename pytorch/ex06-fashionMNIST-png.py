import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

if not os.path.exists('fashionMNIST_images'):
    os.makedirs('fashionMNIST_images/train')
    os.makedirs('fashionMNIST_images/test')

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='vanillagan-fashionmnist/data',
    train=True,
    download=False,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='vanillagan-fashionmnist/data',
    train=False,
    download=False,
    transform=transform
)

def save_images(dataset, dataset_type):
    for index, (image, label) in enumerate(dataset):
        image = transforms.ToPILImage()(image)
        file_name = f'fashionMNIST_images/{dataset_type}/{index}_label{label}.png'
        image.save(file_name)

save_images(train_dataset, 'train')
save_images(test_dataset, 'test')

print("Finished!")
