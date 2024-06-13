import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

training_data = datasets.FashionMNIST(
    root="data/train",
    train=True,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

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

output_dir = "data/images"
if not os.path.exists(output_dir):
     os.makedirs(output_dir)

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
     sample_idx = torch.randint(len(training_data), size=(1,)).item()
     img, label = training_data[sample_idx]
     figure.add_subplot(rows, cols, i)
     plt.title(labels_map[label])
     plt.axis("off")
     plt.imshow(img.squeeze(), cmap="gray")

     img_path = os.path.join(output_dir, f"{labels_map[label]}.png")
     plt.imsave(img_path, img.squeeze(), cmap="gray")
     print(f"Saved image {i} as {img_path}")
plt.show()