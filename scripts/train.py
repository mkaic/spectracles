import torch
from torchvision.datasets import CIFAR100
import torchvision.transforms as tvt
from ..src.model import Spectracles
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import wandb

args = dict(
    mid_layer_size=32,
    num_layers=4,
    n_linear_within_fourier=2,
    normalization_dims=(1, 2, 3),
    residual=True,
    position_embedding_type="sinusoidal",
    position_embedding_size=8,
)


EPOCHS = 50
SAVE = False

print("\n", args)

config = dict(
    **args,
    batch_size=128,
    lr=1e-3,
    data_augmentation=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

if not Path("spectracles/weights").exists():
    Path("spectracles/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

model = Spectracles(num_classes=100, input_channels=3, **args)
model = model.to(DEVICE)
model = model.to(DTYPE)

print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"{num_params:,} trainable parameters")

model = torch.compile(model)

config["num_params"] = num_params

wandb.init(project="spectracles", config=config)

train_transforms = (
    tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=15,
            ),
            tvt.RandomHorizontalFlip(),
            tvt.RandomVerticalFlip(),
            tvt.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
            ),
        ]
    )
    if config["data_augmentation"]
    else tvt.ToTensor()
)

# Load the MNIST dataset
train = CIFAR100(
    root="./spectracles/data", train=True, download=True, transform=train_transforms
)
test = CIFAR100(
    root="./spectracles/data", train=False, download=True, transform=tvt.ToTensor()
)

train_loader = DataLoader(
    train, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=4
)
test_loader = DataLoader(
    test, batch_size=config["batch_size"], shuffle=False, drop_last=True, num_workers=4
)

# Train the model
optimizer = Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.99))

train_accuracy = 0
test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, leave=False)

    total = 0
    correct = 0
    losses = []
    for images, labels in pbar:
        optimizer.zero_grad()

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels = images.to(DTYPE), labels.to(torch.long)

        predictions = model(images)

        predictions = model(images)
        _, predicted = torch.max(predictions, dim=1)

        total += labels.shape[0]
        correct += (predicted == labels).sum().item()

        loss = loss_function(predictions, labels)

        losses.append(loss.item())
        loss.backward()

        optimizer.step()

        pbar.set_description(
            f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Train Acc: {train_accuracy:.2%} | Test Acc: {test_accuracy:.2%}"
        )
    train_accuracy = correct / total

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

    wandb.log(
        {
            "train_loss": torch.tensor(losses).mean(),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    )
