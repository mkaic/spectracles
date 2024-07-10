import torch
import torch.nn as nn
from torch.fft import fft2
from torch import Tensor


# Slightly modified from https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py
class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)
        self.imag_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)

    def forward(self, x: Tensor):

        x = torch.view_as_real(x)
        x_real = x[..., 0]
        x_imag = x[..., 1]

        real = self.real_linear(x_real) - self.imag_linear(x_imag)
        imag = self.imag_linear(x_real) + self.real_linear(x_imag)

        out = torch.stack([real, imag], -1)
        out = torch.view_as_complex(out)

        return out


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)

    def forward(self, x: Tensor):

        x = torch.view_as_real(x)
        x_real = x[..., 0]
        x_imag = x[..., 1]

        # real * real = real, imag * image = -real
        out_real = self.real_conv(x_real) - self.imag_conv(x_imag)
        # real * imag = imag, imag * real = imag
        out_imag = self.imag_conv(x_real) + self.real_conv(x_imag)

        out = torch.stack([out_real, out_imag], -1)
        out = torch.view_as_complex(out)

        return out


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x = torch.view_as_real(x)
        x_real = x[..., 0]
        x_imag = x[..., 1]

        return torch.view_as_complex(
            torch.stack(
                [
                    self.activation(x_real),
                    self.activation(x_imag),
                ],
                dim=-1,
            )
        )


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
        x = fft2(x)
        return x


class ComplexSinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        self.added_channels = num_freqs * 4

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        b, c, h, w = x.shape

        positions = (
            torch.stack(
                torch.meshgrid(
                    *[
                        torch.arange(i, dtype=torch.float32, device=x.device) / (i - 1)
                        for i in (h, w)
                    ],
                    indexing="ij"
                ),
                dim=-1,
            )
            .expand(b, h, w, 2)
            .permute(0, 3, 1, 2)  # B, 2, H, W
        )

        freq_bands = []

        for freq in range(1, self.num_freqs + 1):
            for dim in range(2):
                sin = torch.sin(positions[:, dim] * freq * 2 * torch.pi)
                cos = torch.cos(positions[:, dim] * freq * 2 * torch.pi)
                complex_view = torch.stack([sin, cos], dim=-1)
                complex_view = torch.view_as_complex(complex_view)
                freq_bands.append(complex_view)

        positions = torch.stack(freq_bands, dim=1)

        return torch.cat([x, positions], dim=1)


class FourierBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual,
        n_linear,
        normalization_dims,
        pe_freqs,
    ):

        super().__init__()

        self.residual = residual

        self.layers = [
            Normalization(dims=normalization_dims),
            FourierTransform(),
            ComplexSinusoidalPositionEmbedding2D(num_freqs=pe_freqs),
            ComplexConv2d(
                kernel_size=1,
                in_channels=in_channels + pe_freqs * 2,
                out_channels=in_channels,
            ),
        ]

        for _ in range(n_linear):
            self.layers.extend(
                [
                    ComplexActivation(nn.ReLU()),
                    Normalization(dims=normalization_dims),
                    ComplexSinusoidalPositionEmbedding2D(num_freqs=pe_freqs),
                    ComplexConv2d(
                        kernel_size=1,
                        in_channels=in_channels + pe_freqs * 2,
                        out_channels=in_channels,
                    ),
                ]
            )

        self.layers.extend(
            [
                ComplexActivation(nn.ReLU()),
                Normalization(dims=normalization_dims),
                ComplexSinusoidalPositionEmbedding2D(num_freqs=pe_freqs),
                ComplexConv2d(
                    kernel_size=1,
                    in_channels=in_channels + pe_freqs * 2,
                    out_channels=out_channels,
                ),
            ]
        )

        self.layers = nn.Sequential(*self.layers)

        self.activation = ComplexActivation(nn.ReLU())

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
        i = int(self.relative_coords[0] * (h - 1))
        j = int(self.relative_coords[1] * (w - 1))
        return x[:, :, i, j]


class ComplexAmplitude(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return torch.abs(x)
