from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


REQUIRED_SHARD_FIELDS = (
    "h_ls_lin",
    "h_true",
    "y_rg",
    "x_rg",
    "pilot_mask",
    "snr_db",
    "noise_var",
)


def save_npz_shard(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def validate_shard_payload(payload: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_SHARD_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"Shard payload missing fields: {missing}")

    h_ls_lin = payload["h_ls_lin"]
    h_true = payload["h_true"]
    if h_ls_lin.shape != h_true.shape:
        raise ValueError("h_ls_lin and h_true must share the same shape.")
    if h_ls_lin.ndim != 4 or h_ls_lin.shape[1] < 2:
        raise ValueError("Expected h_ls_lin and h_true to have shape [N, C, S, F] with C >= 2.")


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output
