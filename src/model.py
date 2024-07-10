import torch.nn as nn
from torch import Tensor

from .layers import (
    ComplexAmplitude,
    ComplexLinear,
    FourierBlock,
    SelectPixel,
    AddZeroImagComponent,
)


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        mlp_depth,
        mlp_width,
        residual,
        fourier_channels_proportion=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mlp_width = mlp_width
        self.num_layers = blocks
        self.mlp_depth = mlp_depth
        self.residual = residual
        self.fourier_channels_proportion = fourier_channels_proportion

        self.in_layers = nn.Sequential(
            nn.Conv2d(input_channels, mlp_width, kernel_size=1),
            AddZeroImagComponent(),
        )

        self.mid_layers = nn.Sequential()
        for _ in range(blocks):
            self.mid_layers.append(
                FourierBlock(
                    in_channels=mlp_width,
                    out_channels=mlp_width,
                    residual=residual,
                    n_layers=mlp_depth,
                    fourier_channels_proportion=fourier_channels_proportion,
                )
            )

        self.out_layers = nn.Sequential(
            SelectPixel(relative_coords=(0, 0)),
            ComplexLinear(mlp_width, num_classes),
            ComplexAmplitude(),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = self.in_layers(x)
        x = self.mid_layers(x)
        x = self.out_layers(x)

        return x
