import torch
import torch.nn as nn

from ..src.layers import ComplexMLP, get_rotary_position_vectors, recenter_normalize

from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.io import write_jpeg
from tqdm import tqdm
from pathlib import Path

PE_DIM = 32
WIDTH = 32
DEVICE = torch.device("cuda")
ITERATIONS = 1000
LR = [(0, 0.01)]

if not Path("spectracles/reconstructions").exists():
    Path("spectracles/reconstructions").mkdir(exist_ok=True, parents=True)


class Reconstructor(nn.Module):
    def __init__(self, pe_dim, width):
        super().__init__()
        self.pe_dim = pe_dim
        self.width = width
        self.freq_mlp = ComplexMLP([pe_dim, width, width, width, width], dropout=False)
        self.pixel_mlp = ComplexMLP([pe_dim + width, width, width, width, 3])
        # self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, pos_enc) -> torch.Tensor:
        x = self.freq_mlp(pos_enc)
        x = torch.fft.ifftn(x, dim=(1, 2), norm="ortho")
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.pixel_mlp(x)
        x = torch.abs(x)
        # x = x * self.gamma
        x = 1 - (1 / (1 + x))
        x = x.permute(2, 0, 1)
        return x


image = Image.open("spectracles/branos.jpg").convert("RGB")
image = to_tensor(image)
image = image.to(DEVICE)

c, h, w = image.shape

pos_enc = get_rotary_position_vectors(
    shape=(h, w),
    num_frequencies=PE_DIM // 2,
    device=DEVICE,
)

reconstructor = Reconstructor(PE_DIM, WIDTH).to(DEVICE)
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
