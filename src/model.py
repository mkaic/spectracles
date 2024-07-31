import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    PostFFTBlock,
    PostIFFTBlock,
    ComplexLinear,
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

        self.post_fft_blocks = nn.ModuleList()
        self.post_ifft_blocks = nn.ModuleList()
        for _ in range(blocks):
            self.post_fft_blocks.append(PostFFTBlock(width))
            self.post_ifft_blocks.append(PostIFFTBlock(width))

        self.out_proj = nn.Linear(width, num_classes, bias=True)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        # x = torch.complex(x, torch.zeros_like(x))

        # Bounce back and forth between pixel and frequency spaces
        for i in range(self.num_layers):

            residual = x

            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = self.post_fft_blocks[i](x)

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = x.real

            x = self.post_ifft_blocks[i](x)

            x = x + residual

        # Average all pixels and make final prediction
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        # x = torch.abs(x)

        return x
