import torch.nn as nn
import torch
from torch.fft import fft2
from torch import Tensor
    
class Normalization(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return (x - x.mean(dim=self.dims, keepdim=True)) / (x.std(dim=self.dims, keepdim=True) + 1e-6)

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
    def __init__(self, in_channels, out_channels: int, residual: bool = False):

        super().__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            FourierTransform(),
            Normalization(dims=(1,2,3)),
            SimplePositionEmbedding2D(),
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels * 2 + 2,
                out_channels=out_channels,
            ),
        )
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return self.activation(x + self.layers(x))
        else:
            return self.activation(self.layers(x))


class Spectracles(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        mid_layer_size: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.mid_layer_size = mid_layer_size
        self.num_layers = num_layers

        layers = [FourierBlock(3, mid_layer_size)]
        for i in range(num_layers):
            layers.append(FourierBlock(mid_layer_size, mid_layer_size, residual=True))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(mid_layer_size, num_classes))

        self.sequential = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        return self.sequential(x)
