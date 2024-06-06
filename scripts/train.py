import torch
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from ..src.model import Spectracles
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

NUM_CLASSES = 100
MID_LAYER_SIZE = 64
N_MAIN_LAYERS = 4
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-3
SAVE = False


print(f"{NUM_CLASSES=}, {MID_LAYER_SIZE=}, {N_MAIN_LAYERS=}")

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

print(
    f"""
{BATCH_SIZE=}, 
{EPOCHS=}
"""
)

if not Path("spectracles/weights").exists():
    Path("spectracles/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

model = Spectracles(
    num_classes=NUM_CLASSES,
    mid_layer_size=MID_LAYER_SIZE,
)
model = model.to(DEVICE)
model = model.to(DTYPE)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")


# Load the MNIST dataset
train = CIFAR100(
    root="./spectracles/data", train=True, download=True, transform=ToTensor()
)
test = CIFAR100(
    root="./spectracles/data", train=False, download=True, transform=ToTensor()
)

train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8
)

# Train the model
optimizer = Adam(model.parameters(), lr=LR)

test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    for images, labels in pbar:
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels = images.to(DTYPE), labels.to(torch.long)

        predictions = model(images)

        loss = loss_function(predictions, labels)

        loss.backward()

        optimizer.step()

        pbar.set_description(
            f"Epoch {epoch}. Train: {loss.item():.4f}, Test: {test_accuracy:.2%}"
        )

    model.eval()
    if SAVE:
        torch.save(model.state_dict(), f"spectracles/weights/{epoch:03d}.ckpt")

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, leave=False):

            images: torch.Tensor
            labels: torch.Tensor

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(DTYPE), labels.to(torch.long)

            predictions = model(images)
            _, predicted = torch.max(predictions, dim=1)

            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: {test_accuracy:.2%} accuracy on test set")
