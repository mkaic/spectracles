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

        real_weights = torch.empty((out_features, in_features))
        imag_weights = torch.empty((out_features, in_features))
        nn.init.uniform_(real_weights, -bound, bound)
        nn.init.uniform_(imag_weights, -bound, bound)

        self.weights = torch.complex(real_weights, imag_weights)
        self.weights = nn.Parameter(self.weights)

        if bias:
            real_bias = torch.zeros(out_features)
            imag_bias = torch.zeros(out_features)

            self.biases = torch.complex(real_bias, imag_bias)
            self.biases = nn.Parameter(self.biases)
        else:
            self.register_parameter("biases", None)

    def forward(self, x: torch.Tensor):
        x = F.linear(x, self.weights, self.biases)
        return x


class CompAct(nn.Module):
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
        self.mag_weight_offset = nn.Parameter(torch.zeros((width,)))
        self.mag_bias = nn.Parameter(torch.zeros((width,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_mag = torch.abs(x) + 1e-6
        new_mag = torch.pow(old_mag, self.pow_offset + 1)
        new_mag = new_mag * (self.mag_weight_offset + 1) + self.mag_bias
        x = x * (new_mag / old_mag)
        return x


def maglog(x: torch.Tensor) -> torch.Tensor:
    old_mag = torch.abs(x) + 1e-6
    new_mag = torch.log(old_mag)
    x = x * (new_mag / old_mag)
    return x


class CompExp(nn.Module):
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


class CosRelu(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.cos((torch.pi / 4) - torch.angle(x))


class ComplexMLP(nn.Module):
    def __init__(
        self,
        widths: list[int],
        dropout=False,
    ):

        super().__init__()

        self.layers = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate((zip(widths[:-1], widths[1:]))):

            self.layers.append(ComplexLinear(dim_in, dim_out))

            if i != len(widths) - 2:
                # self.layers.append(CompExp(dim_out))
                self.layers.append(CosRelu())
                if dropout:
                    self.layers.append(ComplexDropout(max_p=0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FourierBlock(nn.Module):
    def __init__(self, width, pe_dim, inverse=False):
        super().__init__()

        self.magnorm_in = MagNorm(width, power=1.0)
        self.magnorm_out = MagNorm(width, power=1.0)
        # self.implicit_filters_in = ComplexMLP([pe_dim+width, width, width], dropout=False)
        self.implicit_filters_out = ComplexMLP(
            [pe_dim + width, width, width], dropout=False
        )

        self.mlp = ComplexMLP([width, width, width], dropout=True)

        self.inverse = inverse

    def forward(self, x: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:

        b, h, w, c = x.shape

        if self.inverse:
            x = torch.fft.ifftn(x, dim=(1, 2), norm="ortho")
        else:
            x = torch.fft.fftn(x, dim=(1, 2), norm="ortho")

        x = recenter_normalize(x)  # mean magnitude is 1

        # x_pe = torch.cat([x, pos_enc.expand(b, -1, -1, -1)], dim=-1)
        # x = x * self.implicit_filters_in(x_pe)

        # x = x * self.implicit_filters_in(pos_enc)
        x = self.magnorm_in(x)

        x = self.mlp(x)

        x = recenter_normalize(x)

        x_pe = torch.cat([x, pos_enc.expand(b, -1, -1, -1)], dim=-1)
        x = x * self.implicit_filters_out(x_pe)

        # x = x * self.implicit_filters_out(pos_enc)

        # x = x * self.implicit_filters_out(pos_enc)
        x = self.magnorm_out(x)

        return x


def complex_grad_clip(grad):
    if grad is not None:
        mag = torch.abs(grad) + 1e-6
        grad = torch.where(mag > 1, grad / mag, grad)
    return grad
