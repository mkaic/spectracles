import torch
import torch.nn as nn
from torch.fft import fft2
from torch import Tensor

class Normalization(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return (x - x.mean(dim=self.dims, keepdim=True)) / (
            x.std(dim=self.dims, keepdim=True) + 1e-6
        )


class FourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape
        x = fft2(x)
        x = torch.view_as_real(x)
        x = x.movedim(-1, 2).contiguous()
        x = x.reshape(b, c * 2, h, w)
        return x


class SimplePositionEmbedding2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.added_channels = 2

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        batch_size, channels, height, width = x.shape

        positions = (
            torch.stack(
                torch.meshgrid(
                    *[
                        torch.arange(i, dtype=x.dtype, device=x.device) / (i - 1)
                        for i in (height, width)
                    ],
                    indexing="ij"
                ),
                dim=-1,
            )
            .expand(batch_size, height, width, 2)
            .permute(0, 3, 1, 2)
        )

        return torch.cat([x, positions], dim=1)


class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        self.added_channels = num_freqs * 4

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        batch_size, channels, height, width = x.shape

        positions = (
            torch.stack(
                torch.meshgrid(
                    *[
                        torch.arange(i, dtype=x.dtype, device=x.device) / (i - 1)
                        for i in (height, width)
                    ],
                    indexing="ij"
                ),
                dim=-1,
            )
            .expand(batch_size, height, width, 2)
            .permute(0, 3, 1, 2)  # B, 2, H, W
        )

        freq_bands = []

        for freq in range(1, self.num_freqs + 1):
            for func in [torch.sin, torch.cos]:
                for dim in range(2):
                    freq_bands.append(func(positions[:, dim] * freq * 2 * torch.pi))

        positions = torch.stack(freq_bands, dim=1)

        return torch.cat([x, positions], dim=1)


class FourierBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        residual,
        n_linear,
        normalization_dims,
        position_embedding,
    ):

        super().__init__()

        self.position_embedding = position_embedding
        position_embedding_chans = position_embedding.added_channels

        self.residual = residual

        self.layers = [
            Normalization(dims=normalization_dims),
            FourierTransform(),
            self.position_embedding,
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels * 2 + position_embedding_chans,
                out_channels=in_channels * 2,
            ),
        ]

        for _ in range(n_linear):
            self.layers.extend(
                [
                    nn.ReLU(),
                    Normalization(dims=normalization_dims),
                    self.position_embedding,
                    nn.Conv2d(
                        kernel_size=1,
                        in_channels=in_channels * 2 + position_embedding_chans,
                        out_channels=in_channels * 2,
                    ),
                ]
            )

        self.layers.extend(
            [
                nn.ReLU(),
                Normalization(dims=normalization_dims),
                self.position_embedding,
                nn.Conv2d(
                    kernel_size=1,
                    in_channels=in_channels * 2 + position_embedding_chans,
                    out_channels=out_channels,
                ),
            ]
        )

        self.layers = nn.Sequential(*self.layers)

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return self.activation(x + self.layers(x))
        else:
            return self.activation(self.layers(x))


class SelectPixel(nn.Module):
    def __init__(self, relative_coords):
        super().__init__()
        self.relative_coords = relative_coords

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape
        i = int(self.relative_coords[0] * (h-1))
        j = int(self.relative_coords[1] * (w-1))
        return x[:, :, i, j]
    
class NoPositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.added_channels = 0

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return x
