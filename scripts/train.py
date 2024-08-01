import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as tvt
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)

import wandb

from ..src.model import Spectracles

warnings.filterwarnings(
    "ignore", "Torchinductor does not support code generation for complex operators"
)

parser = ArgumentParser()
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-p", "--print_params", action="store_true", default=False)
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

model_args = dict(
    blocks=4,
    width=30,
    mlp_depth=2,
    pe_dim=16,
)

config = dict(
    **model_args,
    batch_size=128,
    lr=1e-3,
    data_augmentation=False,
)

EPOCHS = 100
SAVE = True

print(model_args)

if not Path("spectracles/weights").exists():
    Path("spectracles/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

model = Spectracles(num_classes=10, input_channels=3, **model_args)
model = model.to(DEVICE)

# print(model)

num_params = 0
for p in model.parameters():
    if torch.is_complex(p):
        num_params += p.numel() * 2
    else:
        num_params += p.numel()

print(f"{num_params:,} trainable parameters")

if not args.print_params:

    config["num_params"] = num_params

    wandb.init(project="spectracles", config=config, name=args.name)
    include_fn = lambda path: path.endswith(".py")
    wandb.run.log_code("./spectracles", include_fn=include_fn)
    wandb.watch(model, log="parameters", log_freq=390)

    train_transforms = (
        tvt.Compose(
            [
                tvt.RandomAffine(
                    degrees=15,
                    translate=(0.2, 0.2),
                    scale=(0.75, 1.25),
                    shear=10,
                ),
                tvt.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.1,
                ),
                tvt.RandomHorizontalFlip(),
                tvt.RandomVerticalFlip(),
                tvt.ToTensor(),
            ]
        )
        if config["data_augmentation"]
        else tvt.ToTensor()
    )

    # Load the MNIST dataset
    train = CIFAR10(
        root="./spectracles/data", train=True, download=True, transform=train_transforms
    )
    test = CIFAR10(
        root="./spectracles/data", train=False, download=True, transform=tvt.ToTensor()
    )

    train_loader = DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    # Train the model
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    train_accuracy = 0
    test_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, leave=False)

        total = 0
        correct = 0
        losses = []
        for step, (images, labels) in enumerate(pbar):
            optimizer.zero_grad()

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = images.to(torch.float32), labels.to(torch.long)

            predictions = model(images)

            _, predicted = torch.max(predictions, dim=-1)

            if step > len(train_loader) * 0.9:
                total += labels.shape[0]
                correct += (predicted == labels).sum().item()

            loss = loss_function(predictions, labels)

            losses.append(loss.item())
            loss.backward()

            optimizer.step()

            pbar.set_description(
                f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Train Err: {1 - train_accuracy:.2%} | Test Err: {1 - test_accuracy:.2%}"
            )

        train_accuracy = correct / total

        print("\n")
        exp_params = [
            p
            for name, p in model.named_parameters()
            if "mlp" in name and "pow_offset" in name
        ]
        print([f"{p[8].item():.2f}" for p in exp_params])

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
                images, labels = images.to(torch.float32), labels.to(torch.long)

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
