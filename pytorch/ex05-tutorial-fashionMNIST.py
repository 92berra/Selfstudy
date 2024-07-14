import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Lambda

import torch.nn as nn

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    #download=True,
    download=False,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    #download=True,
    download=False,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}") # Feature batch shape: torch.Size([64, 1, 28, 28])
print(f"Labels batch shape: {train_labels.size()}") # Labels batch shape: torch.Size([64])
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}") # Label: 5


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device") # Using mps device


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)

logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")


# Layer:  linear_relu_stack.0.weight | 
#         Size: torch.Size([512, 784]) | 
#         Values: tensor([[-0.0015, -0.0094, -0.0232,  ...,  0.0089, -0.0069,  0.0326], [ 0.0096, -0.0282,  0.0137,  ...,  0.0089,  0.0305,  0.0255]],
#         device='mps:0', grad_fn=<SliceBackward0>)
# Layer: linear_relu_stack.0.bias | 
#         Size: torch.Size([512]) | 
#         Values: tensor([-0.0275,  0.0209], device='mps:0', grad_fn=<SliceBackward0>)
# Layer: linear_relu_stack.2.weight | 
#         Size: torch.Size([512, 512]) | 
#         Values: tensor([[-0.0175,  0.0322, -0.0060,  ..., -0.0274, -0.0021,  0.0386],[ 0.0285,  0.0403, -0.0065,  ..., -0.0134, -0.0122, -0.0394]],
#         device='mps:0', grad_fn=<SliceBackward0>)
# Layer: linear_relu_stack.2.bias | 
#         Size: torch.Size([512]) | 
#         Values: tensor([ 0.0230, -0.0066], device='mps:0', grad_fn=<SliceBackward0>)
# Layer: linear_relu_stack.4.weight | 
#         Size: torch.Size([10, 512]) | 
#         Values: tensor([[ 0.0221,  0.0152,  0.0317,  ...,  0.0048,  0.0117, -0.0415],
#         [-0.0361,  0.0291, -0.0071,  ..., -0.0166,  0.0132, -0.0185]],
#         device='mps:0', grad_fn=<SliceBackward0>)
# Layer: linear_relu_stack.4.bias | 
#         Size: torch.Size([10]) | 
#         Values: tensor([0.0242, 0.0173], device='mps:0', grad_fn=<SliceBackward0>)