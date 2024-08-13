import torch
import torch.nn as nn

from ..src.layers import (
    ComplexMLP,
    get_rotary_position_vectors,
    get_binary_tree_rotary_position_vectors,
)

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.io import write_jpeg
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim

parser = ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

WIDTH = 48
PE_FREQS = 12
DEPTH = 4
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
ITERATIONS = 2000
LR = 5e-2

if not Path("spectracles/reconstructions").exists():
    Path("spectracles/reconstructions").mkdir(exist_ok=True, parents=True)


class Reconstructor(nn.Module):
    def __init__(self, width, depth, pe_dim):
        super().__init__()
        layer_dims = [pe_dim] + [width] * (depth - 1) + [3]
        self.mlp = ComplexMLP(layer_dims)
        # self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, pos_enc) -> torch.Tensor:
        x = self.mlp(pos_enc)
        x = torch.abs(x)
        # x = x * self.gamma
        x = 1 - (1 / (1 + x))
        x = x.permute(2, 0, 1)
        return x


image = Image.open("spectracles/branos.jpg").convert("RGB")
image = to_tensor(image)
# image = torch.rand(3, 256, 256)
write_jpeg((image * 255).to(torch.uint8), "spectracles/reconstructions/original.jpg")
image = image.to(DEVICE)

c, h, w = image.shape

# pos_enc = get_rotary_position_vectors(
#     shape=(h, w),
#     num_frequencies=WIDTH // 2,
#     device=DEVICE,
# )

pos_enc = get_binary_tree_rotary_position_vectors(
    shape=(h, w),
    num_frequencies=PE_FREQS,
    device=DEVICE,
)

reconstructor = Reconstructor(WIDTH, DEPTH, PE_FREQS * 2).to(DEVICE)
optimizer = torch.optim.Adam(reconstructor.parameters(), lr=LR)

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
    optimizer.zero_grad()
    output = reconstructor(pos_enc)
    error = output - image
    ms_ssim_loss = -1 * ms_ssim(
        output.unsqueeze(0), image.unsqueeze(0), data_range=1, size_average=True
    )
    mse = torch.mean(torch.square(error))
    mae = torch.mean(torch.abs(error))
    loss = mse + mae + ms_ssim_loss
    loss.backward()
    optimizer.step()

    pbar.set_description(
        f"RMSE: {torch.sqrt(mse).item():.4f} | MAE: {mae.item():.4f} | SSIM: {-ms_ssim_loss.item():.4f}"
    )

    if i % 100 == 0:
        output = output * 255
        output = output.to("cpu", torch.uint8)
        write_jpeg(output, f"spectracles/reconstructions/{i:04d}.jpg")
        write_jpeg(output, f"spectracles/reconstructions/latest.jpg")

# with torch.no_grad():
#     reconstructor.eval()
#     interpolate_pos_enc = get_binary_tree_rotary_position_vectors(
#         shape=(h * 4, w * 4),
#         num_frequencies=PE_FREQS + 2,
#         device=DEVICE,
#     )
#     interpolate_pos_enc = interpolate_pos_enc[..., 4:]

#     output = reconstructor(interpolate_pos_enc)
#     output = output * 255
#     output = output.to("cpu", torch.uint8)
#     write_jpeg(output, f"spectracles/reconstructions/extrapolated.jpg")
