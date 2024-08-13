import torch.nn as nn
import torch

from .layers import (
    ComplexLinear,
    MagNorm,
    get_rotary_position_vectors,
    get_binary_tree_rotary_position_vectors,
    recenter_normalize,
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
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_layers = blocks
        self.width = width
        self.pe_dim = pe_dim
        self.dropout = dropout

        self.proj_in = nn.Linear(input_channels, width, bias=False)

        self.freq_layers = nn.ModuleList()
        self.pixel_layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.freq_layers.append(
                FourierBlock(
                    width,
                    pe_dim,
                    main_depth=main_mlp_depth,
                    implicit_depth=implicit_mlp_depth,
                    inverse=False,
                    dropout=dropout,
                )
            )
            self.pixel_layers.append(
                FourierBlock(
                    width,
                    pe_dim,
                    main_depth=main_mlp_depth,
                    implicit_depth=implicit_mlp_depth,
                    inverse=True,
                    dropout=dropout,
                ),
            )

        self.out_norm = MagNorm(width, power=1.0)
        self.out_proj = ComplexLinear(width, num_classes, bias=True)

        self.pos_enc = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, h, w = x.shape

        x = torch.movedim(x, 1, -1)  # B, C, H, W -> B, H, W, C
        x = self.proj_in(x)  # increase channel count

        if self.pos_enc is None:
            self.pos_enc = get_binary_tree_rotary_position_vectors(
                shape=x.shape[1:3],
                num_frequencies=(
                    self.pe_dim // 2 if self.pe_dim is not None else x.shape[-1] // 2
                ),
                device=x.device,
            ).unsqueeze(0)

        for i in range(self.num_layers):

            residual = x
            x = self.freq_layers[i](x, self.pos_enc)
            x = self.pixel_layers[i](x, self.pos_enc)
            x = x + residual

        # Average all pixels and make final prediction
        x = self.out_norm(x)
        x = x.mean(dim=(1, 2))
        x = self.out_proj(x)
        x = torch.abs(x)

        return x
