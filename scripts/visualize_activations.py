from pathlib import Path

import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt

from ..src.spectracles import Spectracles

with torch.no_grad():

    DTYPE = torch.float32
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

    epoch = 1

    model_args = dict(
        blocks=12,
        width=12,
        pe_dim=12,
    )

    weights_path = Path("spectracles/weights")

    model = Spectracles(num_classes=10, input_channels=3, **model_args)

    model.load_state_dict(torch.load(weights_path / f"{epoch:03}.ckpt"))

    model = model.to(DEVICE)
    model = model.eval()

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = torch.norm(output[:, :4], dim=-1).cpu()

        return hook

    for name, layer in model.freq_magnorms.named_children():
        layer.register_forward_hook(get_activation(name))

    # Load the CIFAR100 dataset
    test = CIFAR100(
        root="./spectracles/data", train=False, download=True, transform=tvt.ToTensor()
    )
    test_loader = DataLoader(
        test,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    images, labels = next(iter(test_loader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    predictions = model(images)

    _, predicted = torch.max(predictions, dim=1)

    to_plot = {"Original": images.cpu().permute(0, 2, 3, 1).numpy(), **activations}

    # Select the activations for the first 4 images in the batch
    for key, value in activations.items():
        activations[key] = value[:4]

    # Create a grid of plots
    aspect_ratio = len(to_plot) / 4
    fig = plt.figure(figsize=(int(12 * aspect_ratio), 12))
    outer_grid = fig.subfigures(4, len(to_plot))

    for row_idx in range(4):
        for col_idx, (name, tensor) in enumerate(to_plot.items()):
            subfig = outer_grid[row_idx][col_idx]
            if col_idx == 0:
                ax = subfig.subplots()
                ax.imshow(tensor[row_idx])
                ax.axis("off")
                subfig.suptitle("Original")
            else:
                subfig.suptitle(name)
                inner_grid = subfig.subplots(2, 2)

                # Plot the intermediate activations
                for channel_idx, ax in enumerate(inner_grid.flat):
                    ax.imshow(torch.log(tensor[row_idx, channel_idx]).numpy())
                    ax.axis("off")

    plt.savefig("spectracles/activations.png")
