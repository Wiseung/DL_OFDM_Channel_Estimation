from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self
import tomllib


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "tdl_c_siso_baseline"
    seed: int = 1234
    backend: str = "sionna"
    device: str = "auto"


@dataclass(slots=True)
class RadioConfig:
    num_subcarriers: int = 256
    num_ofdm_symbols: int = 14
    pilot_ofdm_symbol_indices: tuple[int, ...] = (2, 11)
    pilot_pattern: str = "kronecker"
    num_tx: int = 1
    num_streams_per_tx: int = 1
    num_tx_ant: int = 1
    num_rx_ant: int = 1
    modulation: str = "qpsk"
    num_bits_per_symbol: int = 2
    channel_model: str = "TDL-C"
    delay_spread_s: float = 100e-9
    speed_m_per_s: float = 10.0
    carrier_frequency_hz: float = 3.5e9
    subcarrier_spacing_hz: float = 15e3


@dataclass(slots=True)
class DatasetConfig:
    output_dir: str = "artifacts/datasets/tdl_c_siso_baseline"
    train_shards: int = 8
    val_shards: int = 2
    test_shards: int = 2
    samples_per_shard: int = 256
    train_snr_min_db: float = 0.0
    train_snr_max_db: float = 20.0
    val_snr_db: float = 10.0
    test_snr_db: float = 10.0


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "cnn"
    in_channels: int = 2
    out_channels: int = 2
    base_channels: int = 64
    num_res_blocks: int = 8
    link_channels: int = 24
    mix_channels: int = 96
    per_link_res_blocks: int = 2
    cross_res_blocks: int = 4
    refine_res_blocks: int = 1


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    early_stopping_patience: int = 8
    num_workers: int = 0
    checkpoint_dir: str = "artifacts/checkpoints/tdl_c_siso_baseline"
    log_every_steps: int = 20


@dataclass(slots=True)
class EvaluationConfig:
    snr_db: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0)
    batches_per_snr: int = 8
    batch_size: int = 64
    output_dir: str = "artifacts/eval/tdl_c_siso_baseline"


@dataclass(slots=True)
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    config_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], config_path: str | Path | None = None) -> Self:
        radio_data = dict(data.get("radio", {}))
        if "pilot_ofdm_symbol_indices" in radio_data:
            radio_data["pilot_ofdm_symbol_indices"] = tuple(radio_data["pilot_ofdm_symbol_indices"])

        evaluation_data = dict(data.get("evaluation", {}))
        if "snr_db" in evaluation_data:
            evaluation_data["snr_db"] = tuple(float(x) for x in evaluation_data["snr_db"])

        return cls(
            experiment=ExperimentConfig(**data.get("experiment", {})),
            radio=RadioConfig(**radio_data),
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**evaluation_data),
            config_path=Path(config_path).resolve() if config_path else None,
        )

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        root = self.config_path.parent if self.config_path else Path.cwd()
        return (root / path).resolve()

    @property
    def dataset_root(self) -> Path:
        return self.resolve_path(self.dataset.output_dir)

    @property
    def checkpoint_root(self) -> Path:
        return self.resolve_path(self.training.checkpoint_dir)

    @property
    def evaluation_root(self) -> Path:
        return self.resolve_path(self.evaluation.output_dir)

    def dataset_split_dir(self, split: str) -> Path:
        return self.dataset_root / split

    def checkpoint_path(self) -> Path:
        return self.checkpoint_root / "best.pt"

    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_root / "latest.pt"

    def metrics_path(self) -> Path:
        return self.checkpoint_root / "metrics.csv"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("config_path", None)
        return data


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    return ProjectConfig.from_mapping(data, config_path=config_path)
