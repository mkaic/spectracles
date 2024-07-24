import torch.nn as nn
from torch import Tensor

from .layers import (
    ComplexAmplitude,
    ComplexLinear,
    FourierAttention,
    MLP,
    ComplexPool,
    ComplexProjection,
    Residual,
    ComplexDropout,
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

        self.in_layers = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=1),
            ComplexProjection(),
        )

        self.mid_layers = nn.Sequential()
        for _ in range(blocks):
            self.mid_layers.extend(
                [
                    Residual(
                        (
                            FourierAttention(width=width),
                            ComplexDropout(0.1),
                            MLP(width),
                        )
                    ),
                ]
            )

        self.out_layers = nn.Sequential(
            ComplexPool(),
            ComplexLinear(width, num_classes),
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
