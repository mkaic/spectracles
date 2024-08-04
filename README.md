
# The idea
In 2021, Google published a paper called [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824). It explores replacing self-attention in a BERT-like model with *just a Fourier transform*. They found that they could achieve *92% the same accuracy* this way. This is a big deal, because it basically gets you `O(n log n)` sequence-length scaling for free. I found this paper thanks to a [wonderful Hacker News commenter](https://news.ycombinator.com/item?id=40515957#40519828). However, I was confused by one choice the authors made (brought to my attention by this [Reddit commenter](https://old.reddit.com/r/MachineLearning/comments/ncdy6m/r_google_replaces_bert_selfattention_with_fourier/gy7hww1/)): they tossed out the imaginary part of the Fourier coefficients! I was suspicious there were gains to be made, so I made this repo. Since I'm not very interested in NLP, I opted to make a vision arch instead.

The architecture treats every pixel as a token.

Interestingly, I have found that this architecture works best if its layers are *complex-valued*!

# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```

# Repo structure
Implementations are in `src`, training script is in `scripts` along with a few sanity-checks. The training script expects CIFAR-10/100 to be in a folder called `data`.