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

        self.rope = RoPE()

        self.proj_in = nn.Linear(input_channels, width * 2)

        self.freq_mlps = nn.ModuleList()
        self.pixel_mlps = nn.ModuleList()
        for _ in range(blocks):
            self.freq_mlps.append(ComplexMLP(width))
            self.pixel_mlps.append(ComplexMLP(width))

        self.out_proj = ComplexLinear(width, num_classes)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        b, h, w, c = x.shape
        x = x.view(b, h, w, c // 2, 2)
        x = torch.view_as_complex(x)
        x = complex_norm(x, dim=(1, 2, 3))

        # Bounce back and forth between pixel and frequency spaces
        for freq_mlp, pixel_mlp in zip(self.freq_mlps, self.pixel_mlps):

            residual = x

            x = complex_norm(x, dim=(1, 2, 3))
            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = self.rope(x)
            x = freq_mlp(x)
            x = complex_norm(x, dim=(1, 2, 3))

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = self.rope(x)
            x = pixel_mlp(x)
            x = x + residual

        # Average all pixels and make final prediction
        x = x.mean(dim=(2, 3))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
