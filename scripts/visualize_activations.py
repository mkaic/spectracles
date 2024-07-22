from pathlib import Path

import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt

from ..src.model import Spectracles

with torch.no_grad():

    DTYPE = torch.float32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    epoch = 1

    args = dict(
        blocks=16,
        mlp_width=32,
        mlp_depth=3,
    )

    config = dict(
        **args,
        batch_size=128,
        lr=1e-3,
        data_augmentation=True,
    )

    weights_path = Path("spectracles/weights")

    model = Spectracles(num_classes=10, input_channels=3, **args)
    model.load_state_dict(torch.load(weights_path / f"{epoch:03}.ckpt"))
    model = model.to(DEVICE)
    model = model.to(DTYPE)
    model = model.eval()

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = torch.norm(output[:, 0], dim=-1).cpu()
        return hook
        
    for name, layer in model.mid_layers.named_children():
        layer.register_forward_hook(get_activation(name))

    # Load the CIFAR100 dataset
    test = CIFAR100(
        root="./spectracles/data", train=False, download=True, transform=tvt.ToTensor()
    )
    test_loader = DataLoader(
        test, batch_size=config["batch_size"], shuffle=False, drop_last=True, num_workers=4
    )

    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    predictions = model(images)

    _, predicted = torch.max(predictions, dim=1)

    # Select the first 4 images in the batch
    images = images[:4]

    # Select the activations for the first 4 images in the batch
    for key, value in activations.items():
        activations[key] = value[:4]

    # Create a grid of plots
    fig, axs = plt.subplots(4, args["blocks"]+1, figsize=(12, 8))

    # Plot the original images
    for i in range(4):
        axs[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axs[i, 0].axis("off")
        axs[i, 0].set_title("Original")

    # Plot the intermediate activations
    for i, (name, value) in enumerate(activations.items()):
        for j in range(4):
            axs[j, i+1].imshow(torch.log(value[j]).numpy())
            axs[j, i+1].axis("off")
            axs[j, i+1].set_title(f"Layer {name}")

    plt.tight_layout()
    plt.savefig("spectracles/activations.png")