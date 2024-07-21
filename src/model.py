import torch.nn as nn
from torch import Tensor

from .layers import (
    ComplexAmplitude,
    ComplexLinear,
    FourierTransform,
    MLP,
    ComplexPool,
    ComplexProjection,
    LayerNorm,
    PixelNorm,
    WeightedResidual,
    ComplexPositionEncoding2D,
    ComplexDropout,
)


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        mlp_depth,
        mlp_width,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mlp_width = mlp_width
        self.num_layers = blocks
        self.mlp_depth = mlp_depth

        self.in_layers = nn.Sequential(
            nn.Conv2d(input_channels, mlp_width, kernel_size=1),
            ComplexProjection(),
        )

        self.mid_layers = nn.Sequential()
        for _ in range(blocks):
            self.mid_layers.extend(
                [
                    WeightedResidual(
                        (
                            LayerNorm(),
                            FourierTransform(dim=(2,3)),
                            ComplexDropout(0.1),
                            ComplexPositionEncoding2D(),
                            MLP(mlp_width, mlp_width, mlp_depth),
                        ),
                        mlp_width,
                    )
                ]
            )

        self.out_layers = nn.Sequential(
            ComplexPool(),
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
