import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = math.sqrt(1 / self.in_features)

        real_weights = torch.empty((out_features, in_features))
        imag_weights = torch.empty((out_features, in_features))
        nn.init.uniform_(real_weights, -bound, bound)
        nn.init.uniform_(imag_weights, -bound, bound)

        self.weights = torch.complex(real_weights, imag_weights)
        self.weights = nn.Parameter(self.weights)

        if bias:
            real_bias = torch.zeros(out_features)
            imag_bias = torch.zeros(out_features)
            # nn.init.uniform_(real_bias, -bound, bound)
            # nn.init.uniform_(imag_bias, -bound, bound)

            self.biases = torch.complex(real_bias, imag_bias)
            self.biases = nn.Parameter(self.biases)
        else:
            self.register_parameter("biases", None)

    def forward(self, x: Tensor):
        x = F.linear(x, self.weights, self.biases)
        return x

class MagLinear(nn.Module):
    def __init__(self, width, bias=True):
        super().__init__()
        self.mag_weight_offset = nn.Parameter(torch.zeros((width,)))
        self.mag_bias = nn.Parameter(torch.zeros((width,)))

    def forward(self, x: Tensor) -> Tensor:
        old_mag = torch.abs(x) + 1e-6
        new_mag = old_mag * (self.mag_weight_offset + 1) + self.mag_bias
        x = x * (new_mag / old_mag)
        return x

class MagAct(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        mag_positivity = torch.cos((torch.pi / 4) - torch.angle(x))
        # mag_positivity = (torch.sign(x.real) + torch.sign(x.imag)) / 2
        # mag_positivity = (mag_positivity + 1) / 2
        mags = torch.abs(x) + 1e-6
        signed_scaled_mags = mags * mag_positivity
        final_mags = self.activation(signed_scaled_mags)
        x = x * (final_mags / mags)

        return x
    

class CompAct(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return torch.complex(self.activation(x.real), self.activation(x.imag))


def recenter(x: Tensor, dim: tuple) -> Tensor:
    return x - torch.mean(x, dim=dim, keepdim=True)



def magnitude_exponent(x: Tensor, pow) -> Tensor:

    mag = torch.abs(x) + 1e-6
    x = x / mag * torch.pow(mag, pow)

    return x


class MagExp(nn.Module):
    def __init__(self, width, pow=1.0):
        super().__init__()
        self.pow_offset = nn.Parameter(torch.full((width,), float(pow - 1)))

    def forward(self, x: Tensor) -> Tensor:
        return magnitude_exponent(x, self.pow_offset + 1)


class MagNorm(nn.Module):
    def __init__(self, width, pow=0.5, recenter=True):
        super().__init__()
        self.magexp = MagExp(width, pow=pow)
        self.maglinear = MagLinear(width, bias=True)
        self.recenter = recenter
    def forward(self, x: Tensor) -> Tensor:
        if self.recenter:
            x = recenter(x, dim=(1,2,3))
        x = self.magexp(x)
        x = self.maglinear(x)
        return x

class CompExp(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.pow_offset = nn.Parameter(torch.zeros((width,), dtype=torch.complex64))

    def forward(self, x: Tensor) -> Tensor:
        return torch.pow(x, self.pow_offset + 1)


def get_rotary_position_vectors(shape, num_frequencies, device):

    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.float32, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    freq_bands = []

    for freq_idx in range(1, num_frequencies + 1):
        for pe_axis in range(2):
            pos = positions[..., pe_axis] * (
                1 / (10000 ** (freq_idx / num_frequencies))
            )
            cos = torch.cos(pos)
            sin = torch.sin(pos)
            complex_view = torch.complex(cos, sin)  # H, W
            freq_bands.append(complex_view)

    positions = torch.stack(freq_bands, dim=-1)  # H, W, C

    return positions


class ComplexMLP(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int = 2,
    ):

        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(ComplexLinear(width, width, bias=True))
            self.layers.append(MagNorm(width, pow=1.0, recenter=False))
            self.layers.append(MagAct(nn.ReLU()))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
