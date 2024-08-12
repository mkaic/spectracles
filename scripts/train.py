import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as tvt
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm
import wandb
from icecream import ic

ic.configureOutput(includeContext=True)

from ..src.model import Spectracles

model_args = dict(
    blocks=8,
    width=32,
    pe_dim=32,
    mlp_depth=4,
    dropout=0.2,
)

config = dict(
    **model_args,
    batch_size=64,
    lr=[(0, 1e-3), (100, 1e-4)],
    data_aug=False,
)


warnings.filterwarnings(
    "ignore", "Torchinductor does not support code generation for complex operators"
)

parser = ArgumentParser()
parser.add_argument("-n", "--name", type=str, default=None)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-p", "--print_params", action="store_true", default=False)
parser.add_argument("-c", "--ckpt", type=str, default=None)
parser.add_argument("-l", "--logs", action="store_true", default=False)
parser.add_argument("-d", "--debug", action="store_true", default=False)
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

if args.debug:
    ic.enable()
    torch.autograd.set_detect_anomaly(True)
else:
    ic.disable()
    torch.autograd.set_detect_anomaly(False)


EPOCHS = 1000
SAVE = True

print(model_args)

if not Path("spectracles/weights").exists():
    Path("spectracles/weights").mkdir(parents=True)

loss_function = nn.CrossEntropyLoss()

model = Spectracles(num_classes=10, input_channels=3, **model_args)
# for p in model.parameters():
#     p.register_hook(complex_grad_clip)

if args.ckpt is not None:
    print(f"Loading weights from {args.ckpt}")
    weights = torch.load(args.ckpt)
    weights.pop("pos_enc")
    print(model.load_state_dict(weights))

model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=config["lr"][0][1], weight_decay=0.01)

if args.print_params:
    print(model)

num_params = 0
for p in model.parameters():
    if torch.is_complex(p):
        num_params += p.numel() * 2
    else:
        num_params += p.numel()

print(f"{num_params:,} trainable parameters")

if not args.print_params:

    config["num_params"] = num_params

    if args.logs:
        wandb.init(project="spectracles", config=config, name=args.name)
        include_fn = lambda path: path.endswith(".py")
        wandb.run.log_code("./spectracles", include_fn=include_fn)
        wandb.watch(model, log="parameters", log_freq=390)

    transform = (
        tvt.Compose(
            [
                tvt.RandomHorizontalFlip(),
                tvt.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                tvt.ToTensor(),
            ]
        )
        if config["data_aug"]
        else tvt.ToTensor()
    )

    train = CIFAR10(
        root="./spectracles/data",
        train=True,
        download=True,
        transform=transform,
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

    train_accuracy = 0
    test_accuracy = 0
    for epoch in range(EPOCHS):

        for e, lr in config["lr"]:
            if epoch == e:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

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

        if args.logs:
            wandb.log(
                {
                    "train_loss": torch.tensor(losses).mean(),
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                }
            )
