# JAX Loop Utils

This repository contains common functionality for writing ML training loops in JAX.
The goal is to make trainings loops short and readable (but moving common tasks to
small libraries) without removing the flexibility required for research.

To get started, check out [this Notebook](./synopsis.ipynb), or just:

```sh
uv add jax-loop-utils
# or if you're not using UV
pip install jax-loop-utils
```

See [pyproject.toml](pyproject.toml) for the optional dependencies, which are
needed for specific metrid writers.

This started as a fork of [CLU](https://github.com/google/CommonLoopUtils).
See [CHANGELOG.md](./CHANGELOG.md) for more details on changes since the fork.
