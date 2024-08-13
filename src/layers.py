import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from icecream import ic


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = math.sqrt(1 / self.in_features)

        weight_magnitudes = torch.empty((out_features, in_features))
        weight_phases = torch.empty((out_features, in_features))

        nn.init.uniform_(weight_magnitudes, 0, bound)
        nn.init.uniform_(weight_phases, -torch.pi, torch.pi)

        self.weights = torch.polar(weight_magnitudes, weight_phases)
        self.weights = nn.Parameter(self.weights)

        if bias:
            bias_magnitudes = torch.empty(out_features)
            bias_phases = torch.empty(out_features)

            nn.init.uniform_(bias_magnitudes, 0, bound)
            nn.init.uniform_(bias_phases, -torch.pi, torch.pi)

            self.biases = torch.polar(bias_magnitudes, bias_phases)
            self.biases = torch.zeros(out_features, dtype=torch.complex64)
            self.biases = nn.Parameter(self.biases)
        else:
            self.register_parameter("biases", None)

    def forward(self, x: torch.Tensor):
        x = F.linear(x, self.weights, self.biases)
        return x


class ComplexActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.complex(self.activation(x.real), self.activation(x.imag))


class MagNorm(nn.Module):
    def __init__(self, width, power=1.0):
        super().__init__()

        self.pow_offset = nn.Parameter(torch.full((width,), float(power - 1)))
        # self.weight_offset = nn.Parameter(torch.zeros((width,)))
        # self.bias = nn.Parameter(torch.zeros((width,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_mag = torch.abs(x) + 1e-6
        new_mag = torch.pow(old_mag, self.pow_offset + 1)
        # new_mag = new_mag * (1 + self.weight_offset) + self.bias
        x = x * (new_mag / old_mag)
        return x


class ComplexPower(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.pow_offset = nn.Parameter(torch.zeros((width,), dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def get_binary_tree_rotary_position_vectors(shape, num_frequencies, device):
    positions = torch.stack(
        torch.meshgrid(
            *[torch.arange(i, dtype=torch.float32, device=device) for i in shape],
            indexing="ij"
        ),
        dim=-1,
    )

    positions = positions * torch.pi  # 1pi, 2pi, 3pi, etc
    freq_bands = []

    for freq_idx in range(num_frequencies):
        for pe_axis in range(2):
            pos = positions[..., pe_axis] / (2**freq_idx)
            freq_bands.append(torch.polar(torch.ones_like(pos), pos))

    positions = torch.stack(freq_bands, dim=-1)  # H, W, C
    return positions


class ComplexDropout(nn.Module):
    def __init__(self, max_p: float = 0.5):
        super().__init__()
        self.max_p = max_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            p = torch.rand(x.shape[0], device=x.device) * self.max_p
            p = p.view(-1, 1, 1, 1)
            mask = torch.rand(x.shape, device=x.device) > p
            x = x * mask
            x = x / (1 - p)
        return x


def recenter_normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - torch.mean(x, dim=(1, 2), keepdim=True)
    x = x / (torch.mean(torch.abs(x), dim=(1, 2), keepdim=True) + 1e-6)
    return x


class LeakyCardioid(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = torch.cos(torch.angle(x)) + 1
        scales = scales * (1 - (self.negative_slope * 2)) + self.negative_slope
        return x * scales


class ComplexMLP(nn.Module):
    def __init__(
        self,
        widths: list[int],
        dropout=None,
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate((zip(widths[:-1], widths[1:]))):

            self.layers.append(ComplexLinear(dim_in, dim_out))

            if i != len(widths) - 2:
                self.layers.append(LeakyCardioid(0.01))
            if dropout:
                self.layers.append(ComplexDropout(max_p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FourierBlock(nn.Module):
    def __init__(
        self, width, pe_dim, main_depth, implicit_depth, inverse=False, dropout=None
    ):
        super().__init__()

        self.magnorm_in = MagNorm(width, power=1.0)
        self.magnorm_out = MagNorm(width, power=1.0)
        self.implicit_filters_out = ComplexMLP(
            [pe_dim] * implicit_depth + [width], dropout=None
        )

        self.mlp = ComplexMLP([width] * (main_depth + 1), dropout=dropout)

        self.inverse = inverse

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:

        if self.inverse:
            x = torch.fft.ifftn(x, dim=(1, 2), norm="ortho")
        else:
            x = torch.fft.fftn(x, dim=(1, 2), norm="ortho")

        x = recenter_normalize(x)

        x = self.magnorm_in(x)

        x = self.mlp(x)

        x = self.implicit_filters_out(pos_enc) * x

        x = self.magnorm_out(x)

        return x


def complex_grad_clip(grad):
    if grad is not None:
        mag = torch.abs(grad) + 1e-6
        grad = torch.where(mag > 1, grad / mag, grad)
    return grad
