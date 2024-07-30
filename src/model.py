import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    ComplexMLP,
    ComplexLinear,
    RealMLP,
    RoPE,
    complex_norm,
    real_norm,
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

        self.proj_in = nn.Linear(input_channels, width, bias=False)

        self.freq_layers_a = nn.ModuleList()
        self.freq_layers_b = nn.ModuleList()
        self.pixel_layers_a = nn.ModuleList()
        self.pixel_layers_b = nn.ModuleList()
        for _ in range(blocks):
            self.freq_layers_a.append(ComplexLinear(width, width))
            self.freq_layers_b.append(ComplexMLP(width))
            self.pixel_layers_a.append(ComplexLinear(width, width))
            self.pixel_layers_b.append(ComplexMLP(width))

        self.out_proj = ComplexLinear(width, num_classes, bias=True)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        x = torch.complex(x, torch.zeros_like(x))

        # Bounce back and forth between pixel and frequency spaces
        for i in range(self.num_layers):

            residual = x

            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = self.freq_layers_a[i](x)
            x = complex_norm(x, dim=(1, 2, 3))
            x = RoPE(x)
            x = self.freq_layers_b[i](x)

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = self.pixel_layers_a[i](x)
            x = complex_norm(x, dim=(1, 2, 3))
            x = RoPE(x)
            x = self.pixel_layers_b[i](x)

            x = x + residual

        # Average all pixels and make final prediction
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
