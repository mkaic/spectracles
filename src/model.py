import torch.nn as nn
from torch import Tensor
from .layers import FourierBlock, SelectPixel, ComplexLinear, ComplexAmplitude


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
        pe_freqs,
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
        self.pe_freqs = pe_freqs

        self.in_layers = FourierBlock(
                3,
                mid_layer_size,
                residual=False,
                n_linear=n_linear_within_fourier,
                normalization_dims=normalization_dims,
                pe_freqs=pe_freqs,
            )
        mid_layers = []
        for _ in range(num_layers):
            mid_layers.append(
                FourierBlock(
                    mid_layer_size,
                    mid_layer_size,
                    residual=residual,
                    n_linear=n_linear_within_fourier,
                    normalization_dims=normalization_dims,
                    pe_freqs=pe_freqs,
                )
            )

        self.mid_layers = nn.Sequential(*mid_layers)

        self.out_layers = nn.Sequential(
            SelectPixel(relative_coords=(0, 0)),
            nn.Flatten(),
            ComplexLinear(mid_layer_size, num_classes),
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
