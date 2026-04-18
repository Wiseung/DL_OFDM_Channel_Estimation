from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .complex_utils import nmse_db, normalized_complex_mse_loss
from .config import ProjectConfig
from .datasets import NpzChannelDataset
from .model import ChannelEstimatorCNN, MIMOFactorizedEstimator
from .sionna_backend import resolve_device


@dataclass(slots=True)
class EpochSummary:
    epoch: int
    train_loss: float
    val_loss: float
    val_nmse_db: float
    throughput_samples_per_s: float


def build_model(config: ProjectConfig, device: torch.device | None = None) -> nn.Module:
    if config.model.architecture == "cnn":
        model = ChannelEstimatorCNN(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            base_channels=config.model.base_channels,
            num_res_blocks=config.model.num_res_blocks,
        )
    elif config.model.architecture == "mimo_factorized":
        num_complex_links = (
            config.radio.num_rx_ant
            * config.radio.num_tx
            * config.radio.num_streams_per_tx
        )
        model = MIMOFactorizedEstimator(
            num_complex_links=num_complex_links,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            link_channels=config.model.link_channels,
            mix_channels=config.model.mix_channels,
            per_link_res_blocks=config.model.per_link_res_blocks,
            cross_res_blocks=config.model.cross_res_blocks,
            refine_res_blocks=config.model.refine_res_blocks,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {config.model.architecture}")

    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_nmse = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            h_ls_lin = batch["input"].to(device)
            delta_true = batch["target"].to(device)
            h_true = batch["h_true"].to(device)
            delta_pred = model(h_ls_lin)
            loss = normalized_complex_mse_loss(delta_pred, delta_true)
            h_pred = h_ls_lin + delta_pred
            total_loss += float(loss.item())
            total_nmse += float(nmse_db(h_pred, h_true).item())
            steps += 1
    if steps == 0:
        raise ValueError("Validation loader produced zero batches.")
    return total_loss / steps, total_nmse / steps


def _summary_to_row(summary: EpochSummary) -> list[str]:
    return [
        str(summary.epoch),
        f"{summary.train_loss:.8f}",
        f"{summary.val_loss:.8f}",
        f"{summary.val_nmse_db:.8f}",
        f"{summary.throughput_samples_per_s:.2f}",
    ]


def _write_metrics(path: Path, summaries: list[EpochSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_nmse_db", "throughput_samples_per_s"])
        for row in summaries:
            writer.writerow(_summary_to_row(row))


def _restore_summaries(checkpoint: dict[str, Any], metrics_path: Path) -> list[EpochSummary]:
    history = checkpoint.get("history")
    if isinstance(history, list) and history:
        return [EpochSummary(**item) for item in history]

    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [
                EpochSummary(
                    epoch=int(row["epoch"]),
                    train_loss=float(row["train_loss"]),
                    val_loss=float(row["val_loss"]),
                    val_nmse_db=float(row["val_nmse_db"]),
                    throughput_samples_per_s=float(row["throughput_samples_per_s"]),
                )
                for row in reader
            ]
    return []


def train_model(config: ProjectConfig, resume_from: str | Path | None = None) -> Path:
    device = resolve_device(config.experiment.device)
    torch.manual_seed(config.experiment.seed)

    train_dataset = NpzChannelDataset(config.dataset_root, split="train")
    val_dataset = NpzChannelDataset(config.dataset_root, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        drop_last=False,
    )

    model = build_model(config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=config.training.amp and device.type == "cuda")
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=config.training.amp and device.type == "cuda")

    config.checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_path()
    checkpoint_path = config.checkpoint_path()
    latest_checkpoint_path = config.latest_checkpoint_path()

    best_val_nmse = float("inf")
    patience = 0
    summaries: list[EpochSummary] = []
    start_epoch = 1

    if resume_from is not None:
        resume_path = Path(resume_from)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)
        best_val_nmse = float(checkpoint.get("best_val_nmse_db", float("inf")))
        patience = int(checkpoint.get("patience", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        summaries = _restore_summaries(checkpoint, metrics_path)

    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        samples_processed = 0
        epoch_start = time.perf_counter()

        for batch in train_loader:
            h_ls_lin = batch["input"].to(device)
            delta_true = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
            with autocast_ctx:
                delta_pred = model(h_ls_lin)
                loss = normalized_complex_mse_loss(delta_pred, delta_true)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            epoch_steps += 1
            samples_processed += h_ls_lin.shape[0]

        val_loss, val_nmse = evaluate_loader(model, val_loader, device=device)
        throughput = samples_processed / max(time.perf_counter() - epoch_start, 1e-6)
        summaries.append(
            EpochSummary(
                epoch=epoch,
                train_loss=epoch_loss / max(epoch_steps, 1),
                val_loss=val_loss,
                val_nmse_db=val_nmse,
                throughput_samples_per_s=throughput,
            )
        )

        improved = val_nmse < best_val_nmse
        if improved:
            best_val_nmse = val_nmse
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "best_val_nmse_db": best_val_nmse,
                    "parameter_count": count_parameters(model),
                    "patience": patience,
                    "history": [asdict(summary) for summary in summaries],
                },
                checkpoint_path,
            )
        else:
            patience += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "config": config.to_dict(),
                "epoch": epoch,
                "best_val_nmse_db": best_val_nmse,
                "parameter_count": count_parameters(model),
                "patience": patience,
                "history": [asdict(summary) for summary in summaries],
                "best_checkpoint_path": str(checkpoint_path),
                "improved_this_epoch": improved,
            },
            latest_checkpoint_path,
        )

        _write_metrics(metrics_path, summaries)

        if patience >= config.training.early_stopping_patience:
            break

    _write_metrics(metrics_path, summaries)

    return checkpoint_path


def load_trained_model(config: ProjectConfig, checkpoint_path: str | Path | None = None) -> tuple[nn.Module, dict]:
    device = resolve_device(config.experiment.device)
    model = build_model(config, device=device)
    checkpoint = torch.load(checkpoint_path or config.checkpoint_path(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
