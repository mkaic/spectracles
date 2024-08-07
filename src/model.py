import torch.nn as nn
import torch

from .layers import (
    CustomSequential,
    ComplexLinear,
    MagExpLin,
    get_rotary_position_vectors,
    FourierBlock,
    Residual,
)


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        width,
        pe_dim=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = blocks
        self.width = width
        self.pe_dim = pe_dim

        self.proj_in = nn.Linear(input_channels, width, bias=False)
        self.magnorm_in = MagExpLin(width, power=1.0)

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(
                Residual(
                    FourierBlock(width, pe_dim, inverse=False),
                    FourierBlock(width, pe_dim, inverse=True),
                )
            )

        self.out_magnorm = MagExpLin(width, power=1.0)
        self.out_proj = ComplexLinear(width, num_classes, bias=True)

        self.pos_enc = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

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

        for layer in self.layers:
            x = layer(x, self.pos_enc)

        # Average all pixels and make final prediction
        x = self.out_magnorm(x)
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
