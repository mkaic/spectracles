import torch.nn as nn
from torch import Tensor
import torch

from .layers import (
    ComplexMLP,
    ComplexLinear,
    MagExpLin,
    get_rotary_position_vectors,
    recenter,
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
        self.mag_exp_in = MagExpLin(width, power=1.0)

        self.post_fft_projs = nn.ModuleList()
        self.freq_mag_exps: list[MagExpLin] = nn.ModuleList()
        self.post_fft_mlps = nn.ModuleList()
        self.post_ifft_projs = nn.ModuleList()
        self.pixel_mag_exps: list[MagExpLin] = nn.ModuleList()
        self.post_ifft_mlps = nn.ModuleList()
        for _ in range(blocks):
            self.post_fft_projs.append(ComplexLinear(width, width, bias=True))
            self.freq_mag_exps.append(MagExpLin(width, power=0.5))
            self.post_fft_mlps.append(ComplexMLP(width, depth=mlp_depth))
            self.post_ifft_projs.append(ComplexLinear(width, width, bias=True))
            self.pixel_mag_exps.append(MagExpLin(width, power=0.5))
            self.post_ifft_mlps.append(ComplexMLP(width, depth=mlp_depth))

        self.out_mag_exp = MagExpLin(width, power=1.0)
        self.out_proj = ComplexLinear(width, num_classes, bias=True)

        self.register_buffer("pos_enc", None)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count
        x = torch.complex(x, torch.zeros_like(x))
        x = recenter(x, dim=(1, 2, 3))
        x = self.mag_exp_in(x)

        if self.pos_enc is None:
            self.pos_enc = get_rotary_position_vectors(
                shape=x.shape[1:3],
                num_frequencies=(
                    self.pe_dim if self.pe_dim is not None else x.shape[-1] // 2
                ),
                device=x.device,
            )

        # Bounce back and forth between pixel and frequency spaces
        for i in range(self.num_layers):

            residual = x

            x = torch.fft.fftn(x, dim=(2, 3), norm="ortho")

            x = self.post_fft_projs[i](x)
            x = recenter(x, dim=(1, 2, 3))
            x = self.freq_mag_exps[i](x)

            x = x * self.pos_enc

            x = self.post_fft_mlps[i](x)

            x = torch.fft.ifftn(x, dim=(2, 3), norm="ortho")

            x = self.post_ifft_projs[i](x)
            x = recenter(x, dim=(1, 2, 3))
            x = self.pixel_mag_exps[i](x)

            x = x * self.pos_enc

            x = self.post_ifft_mlps[i](x)

            x = x + residual

        # Average all pixels and make final prediction
        x = recenter(x, dim=(1, 2, 3))
        x = self.out_mag_exp(x)
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
