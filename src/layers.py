import torch
import torch.nn as nn
from torch import Tensor


# Slightly modified from https://github.com/mehdihosseinimoghadam/Complex-Neural-Networks/blob/main/complex_neural_net.py
class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_linear = nn.Linear(
            self.in_channels, self.out_channels, bias=False, **kwargs
        )
        self.imag_linear = nn.Linear(
            self.in_channels, self.out_channels, bias=False, **kwargs
        )
        self.real_bias = nn.Parameter(torch.zeros(1, self.out_channels))
        self.imag_bias = nn.Parameter(torch.zeros(1, self.out_channels))

    def forward(self, x: Tensor):

        x_real = x[..., 0]
        x_imag = x[..., 1]

        real = self.real_linear(x_real) - self.imag_linear(x_imag) + self.real_bias
        imag = self.imag_linear(x_real) + self.real_linear(x_imag) + self.imag_bias

        out = torch.stack([real, imag], -1)

        return out


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.real_conv = nn.Conv2d(
            self.in_channels, self.out_channels, bias=False, **kwargs
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels, self.out_channels, bias=False, **kwargs
        )
        self.real_bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        self.imag_bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))

    def forward(self, x: Tensor):

        x_real = x[..., 0]
        x_imag = x[..., 1]

        # real * real = real, imag * image = -real
        out_real = self.real_conv(x_real) - self.imag_conv(x_imag) + self.real_bias
        # real * imag = imag, imag * real = imag
        out_imag = self.imag_conv(x_real) + self.real_conv(x_imag) + self.imag_bias

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


def image_norm(x: Tensor) -> Tensor:
    return (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (
        x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
    )


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


class PixelDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        else:
            b, c, h, w, _ = x.shape
            thresholds = torch.rand(b, 1, 1, 1, device=x.device) * self.p
            mask = torch.rand_like(x[..., 0]) > thresholds
            mask = mask.unsqueeze(-1).expand_as(x)

            return x * mask


class MLP(nn.Module):
    def __init__(
        self,
        width: int,
    ):

        super().__init__()

        self.layers = nn.Sequential(
            ComplexPositionEncoding2D(),
            ComplexConv2d(width, width, kernel_size=1, padding=0),
            ComplexActivation(nn.GELU()),
            ComplexConv2d(width, width, kernel_size=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
