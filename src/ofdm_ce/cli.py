from __future__ import annotations

import argparse

from .config import load_config
from .evaluation import evaluate_ber, evaluate_nmse
from .generation import generate_dataset
from .training import train_model


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    return parser


def generate_dataset_main(argv: list[str] | None = None) -> None:
    parser = _base_parser("Generate OFDM dataset shards.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    outputs = generate_dataset(config)
    print(f"Generated {len(outputs)} shards under {config.dataset_root}")


def train_cnn_main(argv: list[str] | None = None) -> None:
    parser = _base_parser("Train the residual CNN channel estimator.")
    parser.add_argument("--resume", help="Optional checkpoint path to resume from.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    checkpoint = train_model(config, resume_from=args.resume)
    print(f"Best checkpoint written to {checkpoint}")


def eval_nmse_main(argv: list[str] | None = None) -> None:
    parser = _base_parser("Evaluate NMSE curves.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    figure = evaluate_nmse(config)
    print(f"NMSE figure written to {figure}")


def eval_ber_main(argv: list[str] | None = None) -> None:
    parser = _base_parser("Evaluate BER curves.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    figure = evaluate_ber(config)
    print(f"BER figure written to {figure}")
