#!/usr/bin/env bash

set -euo pipefail

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null

# Lint
uv run -- ruff format
uv run -- ruff check --fix

# Type check
uv run -- pyright

# Test
uv sync
uv run -- pytest --capture=no --verbose --cov --cov-report=xml \
    --ignore=src/jax_loop_utils/metric_writers/tf/ \
    --ignore=src/jax_loop_utils/metric_writers/torch/ \
    --ignore=src/jax_loop_utils/metric_writers/mlflow/ \
    --ignore=src/jax_loop_utils/metric_writers/_audio_video/ \
    src/jax_loop_utils/

uv sync --extra tensorflow
uv run -- pytest --capture=no --verbose --cov --cov-report=xml --cov-append \
    src/jax_loop_utils/metric_writers/tf

uv sync --group dev-torch --extra torch
uv run -- pytest --capture=no --verbose --cov --cov-report=xml --cov-append \
    src/jax_loop_utils/metric_writers/torch

uv sync --extra mlflow --extra audio-video
uv run -- pytest --capture=no --verbose --cov --cov-report=xml --cov-append \
    src/jax_loop_utils/metric_writers/mlflow

uv sync --extra audio-video
uv run -- pytest --capture=no --verbose --cov --cov-report=xml --cov-append \
    src/jax_loop_utils/metric_writers/_audio_video
