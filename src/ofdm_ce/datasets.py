from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzChannelDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: str | Path, split: str, include_aux: bool = False):
        self.root = Path(root)
        self.split = split
        self.include_aux = include_aux
        self.split_dir = self.root / split
        self.paths = sorted(self.split_dir.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No shard files found under {self.split_dir}")

        self._shard_cache: dict[int, np.lib.npyio.NpzFile] = {}
        self._index: list[tuple[int, int]] = []
        for shard_idx, path in enumerate(self.paths):
            with np.load(path) as shard:
                shard_len = int(shard["h_true"].shape[0])
            self._index.extend((shard_idx, item_idx) for item_idx in range(shard_len))

    def __len__(self) -> int:
        return len(self._index)

    def _tensor(self, value: Any) -> torch.Tensor:
        array = np.asarray(value)
        if array.dtype == np.bool_:
            return torch.from_numpy(array.astype(np.bool_, copy=False))
        return torch.from_numpy(array.astype(np.float32, copy=False))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shard_idx, item_idx = self._index[idx]
        shard = self._shard_cache.get(shard_idx)
        if shard is None:
            shard = np.load(self.paths[shard_idx])
            self._shard_cache[shard_idx] = shard
        h_ls_lin = self._tensor(shard["h_ls_lin"][item_idx])
        h_true = self._tensor(shard["h_true"][item_idx])
        sample = {
            "input": h_ls_lin,
            "target": h_true - h_ls_lin,
            "h_ls_lin": h_ls_lin,
            "h_true": h_true,
            "snr_db": self._tensor(shard["snr_db"][item_idx]),
            "noise_var": self._tensor(shard["noise_var"][item_idx]),
        }
        if self.include_aux:
            for key in ("y_rg", "x_rg", "pilot_mask", "data_mask", "tx_bits"):
                if key in shard:
                    value = shard[key]
                    same_leading_dim = value.ndim > 0 and value.shape[0] == shard["h_true"].shape[0]
                    sample[key] = self._tensor(value[item_idx] if same_leading_dim else value)
        return sample
