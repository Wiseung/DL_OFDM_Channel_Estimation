#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

python -m pip install --upgrade pip setuptools wheel -i "${INDEX_URL}" --trusted-host "${TRUSTED_HOST}"

# Install project-level Python dependencies first.
pip install numpy matplotlib pytest scipy h5py importlib-resources typing-extensions \
  ipywidgets pythreejs filelock sympy networkx jinja2 fsspec \
  -i "${INDEX_URL}" --trusted-host "${TRUSTED_HOST}"

# Install Sionna and Sionna RT packages explicitly to avoid resolver churn.
pip install sionna==2.0.1 sionna-rt==2.0.1 --no-deps \
  -i "${INDEX_URL}" --trusted-host "${TRUSTED_HOST}"

# Install PyTorch and its runtime packages. This can take a while on slow links.
pip install torch==2.9.1 --no-deps \
  -i "${INDEX_URL}" --trusted-host "${TRUSTED_HOST}"
pip install --no-deps \
  nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cuda-runtime-cu12==12.8.90 \
  nvidia-cuda-cupti-cu12==12.8.90 \
  nvidia-cudnn-cu12==9.10.2.21 \
  nvidia-cublas-cu12==12.8.4.1 \
  nvidia-cufft-cu12==11.3.3.83 \
  nvidia-curand-cu12==10.3.9.90 \
  nvidia-cusolver-cu12==11.7.3.90 \
  nvidia-cusparse-cu12==12.5.8.93 \
  nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.27.5 \
  nvidia-nvshmem-cu12==3.3.20 \
  nvidia-nvtx-cu12==12.8.90 \
  nvidia-nvjitlink-cu12==12.8.93 \
  nvidia-cufile-cu12==1.13.1.3 \
  triton==3.5.1 \
  mitsuba==3.8.0 \
  drjit==1.3.1 \
  -i "${INDEX_URL}" --trusted-host "${TRUSTED_HOST}"

# Editable install uses local build tooling already present in the venv.
pip install -e . --no-deps --no-build-isolation

echo "WSL environment is ready."
