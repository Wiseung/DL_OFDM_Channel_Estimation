from pathlib import Path

from ofdm_ce.config import ProjectConfig


def test_config_from_mapping_resolves_paths(tmp_path: Path) -> None:
    config = ProjectConfig.from_mapping(
        {
            "experiment": {"backend": "synthetic"},
            "dataset": {"output_dir": "artifacts/datasets/test"},
            "training": {"checkpoint_dir": "artifacts/checkpoints/test"},
            "evaluation": {"output_dir": "artifacts/eval/test", "snr_db": [0, 10]},
        },
        config_path=tmp_path / "baseline.toml",
    )
    assert config.dataset_root == (tmp_path / "artifacts/datasets/test").resolve()
    assert config.checkpoint_root == (tmp_path / "artifacts/checkpoints/test").resolve()
    assert config.evaluation.snr_db == (0.0, 10.0)

