import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    #download=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    #download=True,
    download=False,
    transform=ToTensor()
)

NOISE_DIM = 10
INPUT_SIZE = 28 * 28
BATCH_SIZE = 64
EPOCHS = 10

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)  # reshape to (batch_size, 1, 28, 28)
        return img
    
generator = Generator(NOISE_DIM).to(device)
print(generator)
print(f"Using {device} device")

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(f"x: {x}")
        print(f"x.shape: {x.shape}") # x.shape: torch.Size([64, 1, 28, 28])
        x = self.model(x)
        return x

discriminator = Discriminator(INPUT_SIZE).to(device)
print(discriminator)

criterion = nn.BCELoss()
optimizer_discriminator = optim.Adam(discriminator.parameters())
optimizer_generator = optim.Adam(discriminator.parameters())

for param in discriminator.parameters():
    param.requires_grad = False

gan_input = torch.randn(BATCH_SIZE, NOISE_DIM).to(device)
fake_images = generator(gan_input)
output = discriminator(fake_images)

def get_batches(data, batch_size):
    batches = []
    for i in range(int(data.shape[0] // batch_size)):
        batch = data[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)
    return np.asarray(batches)