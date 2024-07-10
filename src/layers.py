import torch
import torch.nn as nn
from torch import Tensor
from torch.fft import fft2


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


class AddZeroImagComponent(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([x, torch.zeros_like(x)], dim=-1)


class FourierTransform(nn.Module):
    def __init__(self, proportion_of_channels=None):
        super().__init__()
        self.first_n_channels = proportion_of_channels

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_complex(x):
            x = torch.view_as_complex(x)

        b, c, h, w = x.shape
        if self.first_n_channels is None:
            n = c
        else:
            n = int(self.first_n_channels * c)

        x_ft = x[:, :n]
        x_ft = fft2(x_ft)
        x[:, :n] = x_ft

        x = torch.view_as_real(x)  # B, C, H, W, 2

        return x


class ComplexSinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w, _ = x.shape

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

        num_freqs = c // 2

        for freq in range(1, num_freqs + 1):
            for dim in range(2):
                pos = positions[:, dim] * freq * 2 * torch.pi
                cos = torch.cos(pos)
                sin = torch.sin(pos)
                complex_view = torch.stack([cos, sin], dim=-1)  # B, H, W, 2
                freq_bands.append(complex_view)

        positions = torch.stack(freq_bands, dim=1)  # B, C, H, W, 2

        return x + positions


class SpectraclesLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        do_normalization=True,
        do_activation=True,
        fourier_channels_proportion=None,
        do_fourier=False,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            Normalization(dims=(1, 2, 3)) if do_normalization else nn.Identity(),
            (
                FourierTransform(proportion_of_channels=fourier_channels_proportion)
                if do_fourier
                else nn.Identity()
            ),
            ComplexSinusoidalPositionEmbedding2D(),
            ComplexConv2d(in_channels, out_channels, kernel_size=1, padding=0),
            ComplexActivation(nn.ReLU()) if do_activation else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FourierBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual,
        n_layers,
        fourier_channels_proportion,
    ):

        super().__init__()

        self.residual = residual

        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.append(
                SpectraclesLayer(
                    in_channels=in_channels,
                    out_channels=in_channels if i < n_layers - 1 else out_channels,
                    do_fourier=i == 0,
                    fourier_channels_proportion=fourier_channels_proportion,
                    do_normalization=True,
                    do_activation=i < n_layers - 1,
                )
            )

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

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w, _ = x.shape
        i = int(self.relative_coords[0] * (h - 1))
        j = int(self.relative_coords[1] * (w - 1))
        x = x[:, :, i, j]

        return x


class ComplexAmplitude(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.norm(x, dim=-1)
