from pathlib import Path

from ofdm_ce.config import ProjectConfig
from ofdm_ce.datasets import NpzChannelDataset
from ofdm_ce.generation import generate_dataset
from ofdm_ce.training import train_model


def make_smoke_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig.from_mapping(
        {
            "experiment": {"name": "smoke", "backend": "synthetic", "device": "cpu", "seed": 7},
            "dataset": {
                "output_dir": str(tmp_path / "artifacts/datasets/smoke"),
                "train_shards": 1,
                "val_shards": 1,
                "test_shards": 1,
                "samples_per_shard": 32,
                "train_snr_min_db": 5.0,
                "train_snr_max_db": 5.0,
                "val_snr_db": 5.0,
                "test_snr_db": 5.0,
            },
            "model": {"base_channels": 16, "num_res_blocks": 2},
            "training": {
                "batch_size": 8,
                "epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "amp": False,
                "early_stopping_patience": 2,
                "checkpoint_dir": str(tmp_path / "artifacts/checkpoints/smoke"),
            },
            "evaluation": {"output_dir": str(tmp_path / "artifacts/eval/smoke"), "snr_db": [5.0], "batches_per_snr": 1, "batch_size": 8},
        }
    )


def test_generate_dataset_contract(tmp_path: Path) -> None:
    config = make_smoke_config(tmp_path)
    outputs = generate_dataset(config)
    assert len(outputs) == 3
    dataset = NpzChannelDataset(config.dataset_root, split="train")
    sample = dataset[0]
    assert sample["input"].shape == (2, 14, 256)
    assert sample["h_true"].shape == (2, 14, 256)


def test_training_smoke(tmp_path: Path) -> None:
    config = make_smoke_config(tmp_path)
    generate_dataset(config)
    checkpoint_path = train_model(config)
    assert checkpoint_path.exists()
