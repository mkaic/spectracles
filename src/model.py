import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    ComplexMLP,
    ComplexLinear,
    RealMLP,
    complex_norm,
)


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        width,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.width = width
        self.num_layers = blocks

        self.proj_in = nn.Linear(input_channels, width)

        self.freq_layers = nn.ModuleList()
        self.pixel_layers = nn.ModuleList()
        for _ in range(blocks):
            self.freq_layers.append(ComplexMLP(width))
            self.pixel_layers.append(ComplexMLP(width))

        self.out_proj = ComplexLinear(width, num_classes)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        x = torch.view_as_complex(
            torch.stack([x, torch.zeros_like(x)], dim=-1)
        )  # B, H, W, C -> B, H, W, C
        x = complex_norm(x, dim=(1, 2, 3))

        # Bounce back and forth between pixel and frequency spaces
        for freq_layer, pixel_layer in zip(self.freq_layers, self.pixel_layers):

            residual = x

            x = complex_norm(x, dim=(1, 2, 3))

            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = freq_layer(x)
            x = complex_norm(x, dim=(1, 2, 3))

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = pixel_layer(x)
            x = x + residual

        # Average all pixels and make final prediction
        x = x.mean(dim=(2, 3))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
