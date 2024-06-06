
# The idea


# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```

# Repo structure
Implementations are in `src`, training script is in `scripts`, and sanity-checks I wrote while implementing stuff are in `tests`. `experiments.md` is a log where I track the results of different design choices and hyperparams. The training script expects CIFAR-100 to be in a folder called `data`, which is included in `.gitignore` so I don't accidentally attempt to push the dataset.