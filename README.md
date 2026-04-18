# DL_OFDM_Channel_Estimation

SISO and `2x2 MIMO` OFDM wireless channel estimation baseline repository with four explicit entrypoints:

- `generate_dataset`
- `train_cnn`
- `eval_nmse`
- `eval_ber`

The implementation currently covers:

- SISO `TDL-C`
- SISO `CDL-C`
- `2x2 MIMO` `TDL-C`
- `2x2 MIMO` `CDL-C` smoke-level support in the simulator path

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

The intended runtime is WSL2 Ubuntu with Python 3.11+.

Quick setup:

```bash
cd /mnt/g/OFDM_Channel_Estimation
bash scripts/setup_wsl.sh
source .venv-wsl/bin/activate
pytest -q
```

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

Project write-ups:

- [TDL-C SISO baseline](docs/tdl_c_siso_baseline_results.md)
- [CDL-C SISO baseline](docs/cdl_c_siso_baseline_results.md)
- [2x2 MIMO TDL-C baseline](docs/mimo_2x2_baseline_results.md)

Current headline numbers:

- `TDL-C SISO @ 10 dB`: `LS-lin = -10.45 dB`, `DL = -22.89 dB`
- `CDL-C SISO @ 10 dB`: `LS-lin = -10.45 dB`, `DL = -24.05 dB`
- `2x2 MIMO TDL-C @ 10 dB`: `LS-lin BER = 4.13e-2`, `DL BER = 2.88e-2`

## Notes on Sionna

This repository targets the current official `Sionna 2.0.x / PyTorch` direction. The code keeps Sionna imports behind a lazy boundary so the package remains importable on machines without Sionna, while the validated WSL runtime uses `Sionna 2.0.1`.

If the installed Sionna API differs from the expected `sionna.phy.*` namespace, update the import adapter in `src/ofdm_ce/sionna_backend.py`.
