import torch.nn as nn
import torch
from torch.fft import fft2
from torch import Tensor


class Standardization(nn.Module):
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
        x = x.reshape(b, c * 2, h, w)
        return x


class SimplePositionEmbedding2D(nn.Module):
    def __init__(self):
        super().__init__()

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


class FourierBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        residual,
        n_linear,
        standardization_dims,
        position_embedding_type,
    ):

        super().__init__()
        self.residual = residual
        self.layers = [
            Standardization(dims=standardization_dims),
            FourierTransform(),
            SimplePositionEmbedding2D(),
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels * 2 + 2,
                out_channels=out_channels,
            ),
        ]

        for _ in range(n_linear):
            self.layers.extend(
                [
                    nn.ReLU(),
                    SimplePositionEmbedding2D(),
                    nn.Conv2d(
                        kernel_size=1,
                        in_channels=out_channels + 2,
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
        standardization_dims,
        residual,
        position_embedding_type,
        repetitions=1,
        **kwargs
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mid_layer_size = mid_layer_size
        self.num_layers = num_layers
        self.n_linear_within_fourier = n_linear_within_fourier
        self.standardization_dims = standardization_dims
        self.residual = residual
        self.position_embedding_type = position_embedding_type
        self.repetitions = repetitions

        self.in_layers = nn.Sequential(
            FourierBlock(
                3,
                mid_layer_size,
                residual=False,
                n_linear=n_linear_within_fourier,
                standardization_dims=standardization_dims,
                position_embedding_type=position_embedding_type,
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
                    standardization_dims=standardization_dims,
                    position_embedding_type=position_embedding_type,
                )
            )

        self.main_layers = nn.Sequential(*layers)

        self.out_layers = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(mid_layer_size, num_classes))
        

        

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = self.in_layers(x)

        for i in range(self.repetitions):
            x = self.main_layers(x)

        x = self.out_layers(x)
        
        return x
