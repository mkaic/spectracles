import torch
import matplotlib.pyplot as plt

skewed = torch.randn(100, 2)
skewed[:, 1] = 0.1 * skewed[:, 0] + 0.3 * skewed[:, 1]
plt.scatter(skewed[:, 0], skewed[:, 1], c='r')

# normed = (skewed - skewed.mean(dim=0, keepdim=True)) / skewed.std(dim=0, keepdim=True)
# plt.scatter(normed[:, 0], normed[:, 1], c='b')

c_skewed = torch.view_as_complex(skewed)
c_normed = (c_skewed - c_skewed.mean(dim=0, keepdim=True)) / c_skewed.std(dim=0, keepdim=True)
plt.scatter(c_normed.real, c_normed.imag, c='b')

plt.axis('scaled')
plt.savefig("spectracles/norm.png")