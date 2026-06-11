from pathlib import Path

from ofdm_ce.config import ProjectConfig, load_config


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


def test_mimo_2x2_cdl_c_configs_load() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    smoke = load_config(repo_root / "configs" / "mimo_2x2_cdl_c_smoke.toml")
    wsl = load_config(repo_root / "configs" / "mimo_2x2_cdl_c_wsl_local.toml")

    assert smoke.radio.channel_model == "CDL-C"
    assert smoke.radio.num_tx_ant == 2
    assert smoke.radio.num_rx_ant == 2
    assert smoke.model.in_channels == 8
    assert smoke.dataset_root.name == "mimo_2x2_cdl_c_smoke"

    assert wsl.radio.channel_model == "CDL-C"
    assert "mimo_2x2_cdl_c_baseline" in str(wsl.dataset_root)
    assert "mimo_2x2_cdl_c_baseline" in str(wsl.checkpoint_root)
