import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    ComplexMLP,
    ComplexLinear,
    MagExpLin,
    get_rotary_position_vectors,
)


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        width,
        mlp_depth=2,
        pe_dim=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = blocks
        self.width = width
        self.mlp_depth = mlp_depth
        self.pe_dim = pe_dim

        self.proj_in = nn.Linear(input_channels, width, bias=False)
        self.magnorm_in = MagExpLin(width, power=1.0)

        self.freq_magnorms = nn.ModuleList()
        self.freq_mlps = nn.ModuleList()
        self.freq_magnorms_b = nn.ModuleList()

        self.pixel_magnorms = nn.ModuleList()
        self.pixel_mlps = nn.ModuleList()
        self.pixel_magnorms_b = nn.ModuleList()

        for _ in range(blocks):
            self.freq_magnorms.append(MagExpLin(width, power=0.5))
            self.freq_mlps.append(ComplexMLP(width, width, depth=mlp_depth))

            self.pixel_magnorms.append(MagExpLin(width, power=0.5))
            self.pixel_mlps.append(ComplexMLP(width, width, depth=mlp_depth))
        self.out_magnorm = MagExpLin(width, power=1.0)
        self.out_proj = ComplexLinear(width, num_classes, bias=True)

        self.register_buffer("pos_enc", None)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        b, c, h, w = x.shape

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        x = torch.complex(x, torch.zeros_like(x))
        x = self.magnorm_in(x)

        if self.pos_enc is None:
            self.pos_enc = get_rotary_position_vectors(
                shape=x.shape[1:3],
                num_frequencies=(
                    self.pe_dim // 2 if self.pe_dim is not None else x.shape[-1] // 2
                ),
                device=x.device,
            ).unsqueeze(0)

        # Bounce back and forth between pixel and frequency spaces
        for i in range(self.num_layers):

            residual = x

            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = x * self.pos_enc

            x = self.freq_magnorms[i](x)
            x = self.freq_mlps[i](x)

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = x * self.pos_enc

            x = self.pixel_magnorms[i](x)
            x = self.pixel_mlps[i](x)

            x = x + residual

        # Average all pixels and make final prediction
        x = self.out_magnorm(x)
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
