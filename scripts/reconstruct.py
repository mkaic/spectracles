import torch
import torch.nn as nn

from ..src.layers import ComplexMLP, get_rotary_position_vectors

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.io import write_jpeg
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

WIDTH = 24
DEPTH = 12
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
ITERATIONS = 1000
LR = [(0, 0.001)]

if not Path("spectracles/reconstructions").exists():
    Path("spectracles/reconstructions").mkdir(exist_ok=True, parents=True)


class Reconstructor(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layer_dims = [width] * depth + [3]
        self.mlp = ComplexMLP(layer_dims)

    def forward(self, pos_enc) -> torch.Tensor:
        x = self.mlp(pos_enc)
        x = torch.fft.ifftn(x, dim=(1, 2), norm="backward")
        x = torch.abs(x)
        x = 1 - (1 / (1 + x))
        x = x.permute(2, 0, 1)
        return x


image = Image.open("spectracles/branos.jpg").convert("RGB")
image = to_tensor(image)
image = image.to(DEVICE)

c, h, w = image.shape

pos_enc = get_rotary_position_vectors(
    shape=(h, w),
    num_frequencies=WIDTH // 2,
    device=DEVICE,
)

reconstructor = Reconstructor(WIDTH, DEPTH).to(DEVICE)
optimizer = torch.optim.Adam(reconstructor.parameters(), lr=LR[0][1])

num_params = 0
for p in reconstructor.parameters():
    if torch.is_complex(p):
        num_params += p.numel() * 2
    else:
        num_params += p.numel()

print(f"{num_params:,} trainable parameters")
print(f"{num_params * 4 / 1024:.2f} kB")

pbar = tqdm(range(ITERATIONS + 1))

for i in pbar:
    for step, lr in LR:
        if i == step:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
    optimizer.zero_grad()
    output = reconstructor(pos_enc)
    error = output - image
    mse = torch.mean(torch.square(error))
    mae = torch.mean(torch.abs(error))
    mse.backward()
    optimizer.step()

    pbar.set_description(f"RMSE: {torch.sqrt(mse).item():.4f} | MAE: {mae.item():.4f}")

    if i % 100 == 0:
        output = output * 255
        output = output.to("cpu", torch.uint8)
        write_jpeg(output, f"spectracles/reconstructions/{i:04d}.jpg")
        write_jpeg(output, f"spectracles/reconstructions/latest.jpg")
