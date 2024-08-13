from ..src.layers import get_binary_tree_rotary_position_vectors
import torch

pos_enc = get_binary_tree_rotary_position_vectors(
    shape=(32, 32),
    num_frequencies=5,
    device="cpu",
)

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib


def symmetrical_colormap(cmap, new_name=None):
    """This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold."""
    # get the colormap
    cmap = matplotlib.colormaps.get_cmap(cmap)
    if not new_name:
        new_name = f"sym_{cmap}"  # ex: 'sym_Blues'

    # this defined the roughness of the colormap, 128 fine
    n = 128

    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[
        ::-1
    ]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


mymap = symmetrical_colormap(cmap="plasma", new_name=None)


fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    axs[0][i].imshow(torch.angle(pos_enc[..., i * 2]).cpu().numpy(), cmap=mymap)
    axs[0][i].axis("off")
    axs[1][i].imshow(torch.angle(pos_enc[..., i * 2 + 1]).cpu().numpy(), cmap=mymap)
    axs[1][i].axis("off")

plt.savefig("spectracles/pos_enc.jpg", dpi=600)
