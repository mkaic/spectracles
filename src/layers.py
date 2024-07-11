import torch
import torch.nn as nn
from torch import Tensor
from torch.fft import fft2, fftn


# Slightly modified from https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py
class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)
        self.imag_linear = nn.Linear(self.in_channels, self.out_channels, **kwargs)

    def forward(self, x: Tensor):

        x_real = x[..., 0]
        x_imag = x[..., 1]

        real = self.real_linear(x_real) - self.imag_linear(x_imag)
        imag = self.imag_linear(x_real) + self.real_linear(x_imag)

        out = torch.stack([real, imag], -1)

        return out


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, **kwargs)

    def forward(self, x: Tensor):

        x_real = x[..., 0]
        x_imag = x[..., 1]

        # real * real = real, imag * image = -real
        out_real = self.real_conv(x_real) - self.imag_conv(x_imag)
        # real * imag = imag, imag * real = imag
        out_imag = self.imag_conv(x_real) + self.real_conv(x_imag)

        out = torch.stack([out_real, out_imag], -1)

        return out


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        x_real = x[..., 0]
        x_imag = x[..., 1]

        return torch.stack(
            [
                self.activation(x_real),
                self.activation(x_imag),
            ],
            dim=-1,
        )


class LayerNorm(nn.Module):
    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (
            x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        )


class ComplexProjection(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([x, torch.zeros_like(x)], dim=-1)


class ComplexPool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=(-2, -3))


class FourierTransform(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @torch.compiler.disable()
    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            x = torch.view_as_complex(x.contiguous())

        x = fftn(x, dim=self.dim)

        x = torch.view_as_real(x)  # B, C, H, W, 2

        return x


class ComplexPositionEncoding2D(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w, _ = x.shape

        positions = (
            torch.stack(
                torch.meshgrid(
                    *[
                        torch.arange(i, dtype=torch.float32, device=x.device)
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

        num_freqs = c // 2

        for freq in range(1, num_freqs + 1):
            for dim in range(2):
                pos = positions[:, dim] * (1 / (10000 ** (freq / num_freqs)))
                cos = torch.cos(pos)
                sin = torch.sin(pos)
                complex_view = torch.stack([cos, sin], dim=-1)  # B, H, W, 2
                freq_bands.append(complex_view)

        positions = torch.stack(freq_bands, dim=1)  # B, C, H, W, 2

        return x * positions

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
    ):

        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(n_layers - 1):
            self.layers.extend(
                [
                    LayerNorm(),
                    ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0),
                    ComplexActivation(nn.ReLU()),
                ]
            )
    @torch.compile()
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ComplexAmplitude(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.norm(x, dim=-1)


class Residual(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)
