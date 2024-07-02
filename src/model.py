import torch.nn as nn
import torch
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
        for i in range(num_layers):
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
            nn.AdaptiveAvgPool2d((1, 1)),
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
