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

        self.weights = torch.stack([real_weights, imag_weights], dim=-1)
        self.weights = torch.view_as_complex(self.weights)
        self.weights = nn.Parameter(self.weights)

        if bias:
            real_bias = torch.empty(out_features)
            imag_bias = torch.empty(out_features)
            nn.init.uniform_(real_bias, -bound, bound)
            nn.init.uniform_(imag_bias, -bound, bound)

            self.biases = torch.stack([real_bias, imag_bias], dim=-1)
            self.biases = torch.view_as_complex(self.biases)
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

        x = torch.view_as_complex(torch.stack([real, imag], dim=-1))

        return x


def complex_norm(x: Tensor, dim: tuple) -> Tensor:
    real = (x.real - torch.mean(x.real, dim=dim, keepdim=True)) / (
        torch.std(x.real, dim=dim, keepdim=True) + 1e-6
    )
    imag = (x.imag - torch.mean(x.imag, dim=dim, keepdim=True)) / (
        torch.std(x.imag, dim=dim, keepdim=True) + 1e-6
    )

    x = torch.view_as_complex(torch.stack([real, imag], dim=-1))

    return x


def real_norm(x: Tensor, dim: tuple) -> Tensor:
    return (x - torch.mean(x, dim=dim, keepdim=True)) / (
        torch.std(x, dim=dim, keepdim=True) + 1e-6
    )


class RoPE(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape

        positions = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(i, dtype=torch.float32, device=x.device)
                    for i in (h, w)
                ],
                indexing="ij"
            ),
            dim=-1,
        ).expand(b, h, w, 2)

        freq_bands = []

        num_freqs = c // 2 if torch.is_complex(x) else c // 4

        for freq in range(1, num_freqs + 1):
            for pe_axis in range(2):
                pos = positions[..., pe_axis] * (1 / (10000 ** (freq / num_freqs)))
                cos = torch.cos(pos)
                sin = torch.sin(pos)
                complex_view = torch.view_as_complex(
                    torch.stack([cos, sin], dim=-1)
                )  # B, H, W
                freq_bands.append(complex_view)

        positions = torch.stack(freq_bands, dim=-1)  # B, H, W, C

        if not torch.is_complex(x):
            x = x.view(b, h, w, c // 2, 2)
            x = torch.view_as_complex(x)

        x = x * positions

        if not torch.is_complex(x):
            x = torch.view_as_real(x)
            x = x.view(b, h, w, c)
        
        return x


class ComplexMLP(nn.Module):
    def __init__(
        self,
        width: int,
    ):

        super().__init__()

        self.pe = RoPE()
        self.linear_1 = ComplexLinear(width, width)
        self.activation = ComplexActivation(nn.GELU())
        self.linear_2 = ComplexLinear(width, width)

    def forward(self, x: Tensor) -> Tensor:

        x = self.pe(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x


class RealMLP(nn.Module):
    def __init__(
        self,
        width: int,
    ):

        super().__init__()

        self.linear_1 = nn.Linear(width, width, kernel_size=1, padding=0)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(width, width, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)

        return x
