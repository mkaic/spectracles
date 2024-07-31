import torch
import matplotlib.pyplot as plt

skewed = torch.randn(50, 2)
skewed[:, 1] = 0.1 * skewed[:, 0] + 0.3 * skewed[:, 1] + 0.5
skewed = skewed + torch.tensor([0.0, 2.5])
plt.scatter(skewed[:, 0], skewed[:, 1], c="r", s=2)

# normed = (skewed - skewed.mean(dim=0, keepdim=True)) / skewed.std(dim=0, keepdim=True)
# plt.scatter(normed[:, 0], normed[:, 1], c='b')

c_skewed = torch.view_as_complex(skewed)

c_normed = c_skewed - c_skewed.mean(dim=0, keepdim=True)
mag = torch.abs(c_normed) + 1e-6
c_normed_2 = c_normed / mag * torch.pow(mag, 1 / 2)
c_normed_3 = c_normed / mag * torch.pow(mag, 1 / 3)
c_normed_4 = c_normed / mag * torch.pow(mag, 1 / 4)

# c_normed = (c_skewed - c_skewed.mean(dim=0, keepdim=True)) / c_skewed.std(
#     dim=0, keepdim=True
# )
plt.scatter(c_normed_2.real, c_normed_2.imag, c="b", s=2)
plt.scatter(c_normed_3.real, c_normed_3.imag, c="g", s=2)
plt.scatter(c_normed_4.real, c_normed_4.imag, c="black", s=2)

plt.axis("scaled")
plt.xlim(-3, 3)
plt.ylim(-1, 4)
plt.savefig("spectracles/norm.png")
