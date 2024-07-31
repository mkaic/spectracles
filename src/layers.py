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


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        real = self.activation(x.real)
        imag = self.activation(x.imag)

        x = torch.complex(real, imag)

        return x


def recenter(x: Tensor, dim: tuple) -> Tensor:
    return x - torch.mean(x, dim=dim, keepdim=True)


class Recenter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return recenter(x, self.dim)


def magnitude_exponent(x: Tensor, pow) -> Tensor:

    mag = torch.abs(x) + 1e-6
    x = x / mag * torch.pow(mag, pow)

    return x


class MagnitudeExponent(nn.Module):
    def __init__(self, width, pow=1.0):
        super().__init__()
        self.pow_offset = nn.Parameter(torch.full((width,), float(pow - 1)))

    def forward(self, x: Tensor) -> Tensor:
        return magnitude_exponent(x, self.pow_offset + 1)


class ComplexExponential(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.powers = nn.Parameter(torch.ones((width,), dtype=torch.complex64))

    def forward(self, x: Tensor) -> Tensor:
        return torch.pow(x, self.powers)


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
            # self.layers.append(Recenter(dim=-1))
            self.layers.append(ComplexExponential(width))

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
