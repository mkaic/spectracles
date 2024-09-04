import torch.nn as nn
import torch

from .layers import (
    ComplexLinear,
    MagNorm,
    get_rotary_position_vectors,
    get_binary_tree_rotary_position_vectors,
    FourierBlock,
)

from icecream import ic


class Spectracles(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        blocks,
        width,
        main_mlp_depth,
        implicit_mlp_depth,
        pe_dim=None,
        dropout=None,
        autoencoder=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = blocks
        self.width = width
        self.pe_dim = pe_dim
        self.dropout = dropout
        self.autoencoder = autoencoder

        self.proj_in = ComplexLinear(input_channels, width, bias=True)

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.layers.append(
                FourierBlock(
                    width,
                    pe_dim,
                    main_depth=main_mlp_depth,
                    implicit_depth=implicit_mlp_depth,
                    dropout=dropout,
                )
            )

        self.out_norm = MagNorm(width, power=1.0)
        self.out_proj = ComplexLinear(width, num_classes, bias=True)

        self.pos_enc = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = torch.complex(x, torch.zeros_like(x))

        x = self.proj_in(x)  # increase channel count
        _, _, _, c = x.shape

        x[..., c // 2 :] = torch.fft.fftn(x[..., c // 2 :], dim=(1, 2), norm="ortho")

        if self.pos_enc is None:
            self.pos_enc = get_rotary_position_vectors(
                shape=x.shape[1:3],
                num_frequencies=(
                    self.pe_dim // 2 if self.pe_dim is not None else x.shape[-1] // 2
                ),
                device=x.device,
            ).unsqueeze(0)

        for i in range(self.num_layers):

            x = self.layers[i](x, self.pos_enc)

        # Average all pixels and make final prediction
        x = self.out_norm(x)

        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
