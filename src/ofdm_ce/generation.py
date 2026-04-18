from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np

from .config import ProjectConfig
from .io import save_npz_shard, validate_shard_payload, write_json
from .sionna_backend import SimulatorBatch, create_simulator


def batch_to_payload(batch: SimulatorBatch) -> dict[str, np.ndarray]:
    payload = {
        "h_ls_lin": batch.h_ls_lin.numpy(),
        "h_true": batch.h_true.numpy(),
        "y_rg": batch.y_rg.numpy(),
        "x_rg": batch.x_rg.numpy(),
        "pilot_mask": batch.pilot_mask.numpy(),
        "data_mask": batch.data_mask.numpy(),
        "snr_db": batch.snr_db.numpy(),
        "noise_var": batch.noise_var.numpy(),
        "tx_bits": batch.tx_bits.numpy(),
        "h_ls_nn": batch.h_ls_nn.numpy(),
        "h_ls_lin_time_avg": batch.h_ls_lin_time_avg.numpy(),
        "h_lmmse": batch.h_lmmse.numpy(),
    }
    if batch.tx_bits_data is not None:
        payload["tx_bits_data"] = batch.tx_bits_data.numpy()
    validate_shard_payload(payload)
    return payload


def generate_dataset(config: ProjectConfig) -> list[Path]:
    simulator = create_simulator(config)
    outputs: list[Path] = []

    split_plan = {
        "train": (config.dataset.train_shards, None),
        "val": (config.dataset.val_shards, config.dataset.val_snr_db),
        "test": (config.dataset.test_shards, config.dataset.test_snr_db),
    }

    metadata = {
        "experiment": config.experiment.name,
        "backend": config.experiment.backend,
        "dataset": asdict(config.dataset),
        "radio": asdict(config.radio),
    }
    write_json(config.dataset_root / "metadata.json", metadata)

    for split, (num_shards, split_snr) in split_plan.items():
        split_dir = config.dataset_split_dir(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(num_shards):
            batch = simulator.generate_batch(config.dataset.samples_per_shard, snr_db=split_snr)
            payload = batch_to_payload(batch)
            shard_path = split_dir / f"{split}_{shard_idx:04d}.npz"
            outputs.append(save_npz_shard(shard_path, payload))

    return outputs
