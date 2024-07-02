import torch.nn as nn
import torch
from torch import Tensor
from .layers import *

class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        mid_layer_size,
        num_layers,
        n_linear_within_fourier,
        normalization_dims,
        residual,
        position_embedding_type,
        position_embedding_size,
        **kwargs
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mid_layer_size = mid_layer_size
        self.num_layers = num_layers
        self.n_linear_within_fourier = n_linear_within_fourier
        self.normalization_dims = normalization_dims
        self.residual = residual

        self.position_embedding = {
            "simple": SimplePositionEmbedding2D(),
            "sinusoidal": SinusoidalPositionEmbedding2D(position_embedding_size),
            "none": NoPositionEmbedding(),
        }[position_embedding_type]

        self.in_layers = nn.Sequential(
            FourierBlock(
                3,
                mid_layer_size,
                residual=False,
                n_linear=n_linear_within_fourier,
                normalization_dims=normalization_dims,
                position_embedding=self.position_embedding,
            )
        )
        layers = []
        for _ in range(num_layers):
            layers.append(
                FourierBlock(
                    mid_layer_size,
                    mid_layer_size,
                    residual=residual,
                    n_linear=n_linear_within_fourier,
                    normalization_dims=normalization_dims,
                    position_embedding=self.position_embedding,
                )
            )

        self.main_layers = nn.Sequential(*layers)

        self.out_layers = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            SelectPixel(relative_coords=(0,0)),
            nn.Flatten(),
            nn.Linear(mid_layer_size, num_classes),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = self.in_layers(x)

        x = self.main_layers(x)

        x = self.out_layers(x)

        return x
