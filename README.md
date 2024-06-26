
# The idea
In 2021, a paper was published by Google researchers called [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824). In it, they explore the possibility of replacing self-attention in a BERT-like language model with literally *just a Fourier transform*. And they found that they could achieve *92% the same accuracy* this way while getting `O(n log n)` sequence-length scaling for practically free. I only found out about this paper last week thanks to this [wonderful Hacker News commenter](https://news.ycombinator.com/item?id=40515957#40519828), but it's been on my mind ever since. Using the Fourier Transform to allow global information mixing makes *so much sense*. I was a bit perplexed by one choice the authors made, though (brought to my attention by this [Reddit commenter](https://old.reddit.com/r/MachineLearning/comments/ncdy6m/r_google_replaces_bert_selfattention_with_fourier/gy7hww1/)): they tossed out the imaginary part of the resulting coefficients! That seemed odd to me.

In this repo, I'm investigating an extremely simple and computationally efficient image classification architecture which alternates between convolutional layers with kernel-size 1 (equivalent to running a linear layer on every pixel) and Fourier Transform layers which produce a grid of complex-valued coefficients the same size as the input image. I take these complex coefficients and "real-ify" them, effectively doubling the number of channels. I also concatenate stupid-simple "positional embeddings" as an extra two channels after every Fourier layer.

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```

# Repo structure
Implementations are in `src`, training script is in `scripts`, and sanity-checks I wrote while implementing stuff are in `tests`. The training script expects CIFAR-100 to be in a folder called `data`, which is included in `.gitignore` so I don't accidentally attempt to push the dataset.