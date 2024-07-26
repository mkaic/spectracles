import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    MLP,
    PixelDropout,
    ComplexLinear,
    image_norm,
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

        self.proj_in = nn.Conv2d(input_channels, width, kernel_size=1, bias=False)

        self.pixel_dropout = PixelDropout(p=1.0)

        self.freq_layers = nn.ModuleList()
        self.pixel_layers = nn.ModuleList()
        for _ in range(blocks):
            self.freq_layers.append(MLP(width))
            self.pixel_layers.append(MLP(width))

        self.out_proj = ComplexLinear(width, num_classes)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = self.proj_in(x)  # increase channel count
        x = image_norm(x)
        x = torch.stack([x, torch.zeros_like(x)], dim=-1)  # make "complex"

        pixel_residual = x

        # Bounce back and forth between pixel and frequency spaces
        for freq_layer, pixel_layer in zip(self.freq_layers, self.pixel_layers):

            x = image_norm(x)

            x = torch.view_as_complex(x.contiguous())
            x = torch.fft.fftn(x, dim=(1, 2, 3), norm="ortho")
            x = torch.view_as_real(x)  # B, C, H, W, 2

            x = self.pixel_dropout(x)

            x = freq_layer(x)

            x = image_norm(x)

            x = torch.view_as_complex(x.contiguous())
            x = torch.fft.ifftn(x, dim=(1, 2, 3), norm="ortho")
            x = torch.view_as_real(x)  # B, C, H, W, 2

            x = pixel_layer(x)
            x = x + pixel_residual

            pixel_residual = x

        # Average all pixels and make final prediction
        x = x.mean(dim=(2, 3))
        x = self.out_proj(x)
        x = torch.norm(x, dim=-1)

        return x
