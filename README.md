# OFDM Channel Estimation Baseline

SISO OFDM wireless channel estimation baseline repository with four explicit entrypoints:

- `generate_dataset`
- `train_cnn`
- `eval_nmse`
- `eval_ber`

The implementation follows the requested first milestone:

- SISO
- 14 OFDM symbols
- 256 subcarriers
- pilot OFDM symbol indices `{2, 11}`
- QPSK
- fixed `10 m/s` baseline channel condition
- CNN residual refinement on top of `LS-lin`

## Repository Layout

```text
configs/
  baseline.toml
  baseline_wsl_local.toml
  cdl_c_siso_smoke.toml
  cdl_c_siso_wsl_local.toml
  mimo_2x2_smoke.toml
  mimo_2x2_wsl_local.toml
scripts/
  setup_wsl.sh
src/ofdm_ce/
  cli.py
  config.py
  complex_utils.py
  datasets.py
  evaluation.py
  generation.py
  io.py
  model.py
  sionna_backend.py
  training.py
tests/
```

## Environment

The host Windows Python in this workspace is not the target runtime. Use WSL2 Ubuntu with Python 3.11+.

Quick setup:

```bash
cd /mnt/g/OFDM_Channel_Estimation
bash scripts/setup_wsl.sh
source .venv-wsl/bin/activate
pytest -q
```

The implementation keeps the Sionna dependency optional at import time so the repository stays testable on machines that do not yet have Sionna installed. For local smoke tests and lightweight checks, a synthetic backend is included.

If your WSL environment has SSL issues against `pypi.org`, the setup script defaults to the Tsinghua mirror. You can override it with:

```bash
PIP_INDEX_URL=https://your-mirror/simple PIP_TRUSTED_HOST=your-mirror-host bash scripts/setup_wsl.sh
```

## Default Config

The baseline configuration is in `configs/baseline.toml`.

Important defaults:

- `backend = "sionna"` for real dataset generation and evaluation
- `backend = "synthetic"` for smoke tests without Sionna
- training input is `h_ls_lin`
- training target is `h_true - h_ls_lin`
- stored shard tensors use `[N, C, S, F]`; for SISO, `C=2`, and for `2x2 MIMO`, `C=8`

## Entrypoints

```bash
generate_dataset --config configs/baseline.toml
train_cnn --config configs/baseline.toml
eval_nmse --config configs/baseline.toml
eval_ber --config configs/baseline.toml
```

Resume training from an existing checkpoint:

```bash
train_cnn --config configs/baseline_wsl_local.toml --resume /path/to/checkpoints/latest.pt
```

## Outputs

By default the repository writes to:

- `artifacts/datasets/<experiment>/`
- `artifacts/checkpoints/<experiment>/`
- `artifacts/eval/<experiment>/`

Artifacts are intentionally ignored by git.

## Results

The current validated baseline write-up is in [docs/tdl_c_siso_baseline_results.md](docs/tdl_c_siso_baseline_results.md).
The corresponding `CDL-C` write-up is in [docs/cdl_c_siso_baseline_results.md](docs/cdl_c_siso_baseline_results.md).
The `2x2 MIMO` write-up is in [docs/mimo_2x2_baseline_results.md](docs/mimo_2x2_baseline_results.md).

Key points from the latest real-backend run:

- `Sionna 2.0.1 + PyTorch 2.9.1` validated on `RTX 4060 Laptop GPU`
- best checkpoint currently saved at `epoch 17`
- native WSL continuation confirmed the early-stopping plateau at `epoch 25`
- `NMSE @ 10 dB`: `LS-lin = -10.45 dB`, `DL = -22.89 dB`
- `BER @ 10 dB`: `LS-lin = 3.12e-2`, `DL = 1.46e-2`

## Notes on Sionna

This repository is implemented against the current official Sionna/PyTorch direction as of `2026-04-16`. The code still keeps the Sionna import behind a lazy boundary so the package remains testable on machines that do not have Sionna installed, but the WSL runtime used for validation in this workspace already has `Sionna 2.0.1` installed and verified.

If the installed Sionna API differs from the expected `sionna.phy.*` namespace, update the import adapter in `src/ofdm_ce/sionna_backend.py`.
