from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from .complex_utils import channels_to_complex_tensor, nmse_linear
from .config import ProjectConfig
from .sionna_backend import create_simulator, qpsk_demapper, resolve_device
from .training import load_trained_model


def _supports_siso_ber(config: ProjectConfig) -> bool:
    return (
        config.radio.num_tx == 1
        and config.radio.num_streams_per_tx == 1
        and config.radio.num_tx_ant == 1
        and config.radio.num_rx_ant == 1
    )


def _supports_mimo_ber(config: ProjectConfig) -> bool:
    return config.experiment.backend == "sionna" and (
        config.radio.num_streams_per_tx > 1
        or config.radio.num_rx_ant > 1
        or config.radio.num_tx_ant > 1
    )


def _collect_nmse_results(config: ProjectConfig) -> list[dict[str, float]]:
    device = resolve_device(config.experiment.device)
    simulator = create_simulator(config)
    model, _ = load_trained_model(config)
    results: list[dict[str, float]] = []

    for snr_db in config.evaluation.snr_db:
        metrics = {"ls_nn": 0.0, "ls_lin": 0.0, "ls_lin_time_avg": 0.0, "lmmse": 0.0, "dl": 0.0}
        for _ in range(config.evaluation.batches_per_snr):
            batch = simulator.generate_batch(config.evaluation.batch_size, snr_db=snr_db)
            h_true = batch.h_true.to(device)
            h_ls_nn = batch.h_ls_nn.to(device)
            h_ls_lin = batch.h_ls_lin.to(device)
            h_ls_lin_time_avg = batch.h_ls_lin_time_avg.to(device)
            h_lmmse = batch.h_lmmse.to(device)

            with torch.no_grad():
                h_dl = model.predict_channel(h_ls_lin)

            metrics["ls_nn"] += float(nmse_linear(h_ls_nn, h_true).item())
            metrics["ls_lin"] += float(nmse_linear(h_ls_lin, h_true).item())
            metrics["ls_lin_time_avg"] += float(nmse_linear(h_ls_lin_time_avg, h_true).item())
            metrics["lmmse"] += float(nmse_linear(h_lmmse, h_true).item())
            metrics["dl"] += float(nmse_linear(h_dl, h_true).item())

        result = {"snr_db": float(snr_db)}
        for name, value in metrics.items():
            result[name] = 10.0 * torch.log10(torch.tensor(value / config.evaluation.batches_per_snr)).item()
        results.append(result)
    return results


def evaluate_nmse(config: ProjectConfig) -> Path:
    output_dir = config.evaluation_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nmse_results.csv"
    fig_path = output_dir / "nmse_curve.png"

    results = _collect_nmse_results(config)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    snrs = [item["snr_db"] for item in results]
    plt.figure(figsize=(8, 5))
    for name, label in (
        ("ls_nn", "LS-nn"),
        ("ls_lin", "LS-lin"),
        ("ls_lin_time_avg", "LS-lin_time_avg"),
        ("lmmse", "LMMSE-like"),
        ("dl", "DL"),
    ):
        plt.plot(snrs, [item[name] for item in results], marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("NMSE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path


def _one_tap_lmmse_equalize(y_rg: torch.Tensor, h_est: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    y_complex = channels_to_complex_tensor(y_rg)
    h_complex = channels_to_complex_tensor(h_est)
    denom = h_complex.abs().pow(2) + noise_var.view(-1, 1, 1)
    return torch.conj(h_complex) * y_complex / denom.clamp_min(1e-8)


def _ber_for_estimate(y_rg: torch.Tensor, h_est: torch.Tensor, tx_bits: torch.Tensor, data_mask: torch.Tensor, noise_var: torch.Tensor) -> float:
    equalized = _one_tap_lmmse_equalize(y_rg, h_est, noise_var)
    estimated_bits = qpsk_demapper(equalized).cpu()
    mask = data_mask.bool()
    target = tx_bits[:, mask, :].reshape(-1, 2)
    pred = estimated_bits[:, mask, :].reshape(-1, 2)
    bit_errors = (pred != target).float().sum().item()
    total_bits = float(target.numel())
    return bit_errors / max(total_bits, 1.0)


def _ber_for_estimate_sionna(
    equalized_grid: torch.Tensor,
    tx_bits_data: torch.Tensor,
    noise_var: torch.Tensor,
    resource_grid_demapper,
    bit_demapper,
) -> float:
    demapper_device = torch.device(bit_demapper.device)
    equalized_grid_complex = equalized_grid.to(demapper_device).unsqueeze(1).unsqueeze(1)
    y_data = resource_grid_demapper(equalized_grid_complex)
    hard_bits = bit_demapper(y_data, noise_var.to(demapper_device).view(-1, 1, 1, 1))
    hard_bits = hard_bits.reshape(hard_bits.shape[0], -1, 2).cpu()
    target_bits = tx_bits_data.reshape(tx_bits_data.shape[0], -1, 2).cpu()
    bit_errors = (hard_bits != target_bits).float().sum().item()
    total_bits = float(target_bits.numel())
    return bit_errors / max(total_bits, 1.0)


def _ber_for_estimate_sionna_mimo(
    simulator,
    h_est_flat: torch.Tensor,
    y_rg_flat: torch.Tensor,
    tx_bits_data: torch.Tensor,
    noise_var: torch.Tensor,
    equalizer,
    bit_demapper,
) -> float:
    equalizer_device = torch.device(equalizer.device)
    h_est = simulator.unflatten_h(h_est_flat.to(equalizer_device))
    y_rg = simulator.unflatten_y(y_rg_flat.to(equalizer_device))
    err_var = torch.zeros_like(h_est.real)
    no = noise_var.to(equalizer_device).view(-1, 1, 1)
    x_hat, no_eff = equalizer(y_rg, h_est, err_var, no)
    hard_bits = bit_demapper(x_hat, no_eff)
    num_data_symbols = simulator.rg.num_data_symbols
    hard_bits = hard_bits.reshape(
        hard_bits.shape[0],
        simulator.config.radio.num_tx,
        simulator.config.radio.num_streams_per_tx,
        num_data_symbols,
        simulator.config.radio.num_bits_per_symbol,
    ).cpu()
    target_bits = tx_bits_data.reshape(
        tx_bits_data.shape[0],
        simulator.config.radio.num_tx,
        simulator.config.radio.num_streams_per_tx,
        num_data_symbols,
        simulator.config.radio.num_bits_per_symbol,
    ).cpu()
    bit_errors = (hard_bits != target_bits).float().sum().item()
    total_bits = float(target_bits.numel())
    return bit_errors / max(total_bits, 1.0)


def evaluate_ber(config: ProjectConfig) -> Path:
    device = resolve_device(config.experiment.device)
    simulator = create_simulator(config)
    model, _ = load_trained_model(config)
    resource_grid_demapper = None
    bit_demapper = None
    mimo_equalizer = None

    if config.experiment.backend == "sionna":
        from sionna.phy.mapping import Demapper
        from sionna.phy.mimo import StreamManagement
        from sionna.phy.ofdm import LMMSEEqualizer, ResourceGridDemapper

        stream_management = StreamManagement(np.ones((1, config.radio.num_tx), dtype=np.int32), config.radio.num_streams_per_tx)
        bit_demapper = Demapper("maxlog", "qam", config.radio.num_bits_per_symbol, hard_out=True)
        if _supports_siso_ber(config):
            resource_grid_demapper = ResourceGridDemapper(simulator.rg, stream_management)
        elif _supports_mimo_ber(config):
            mimo_equalizer = LMMSEEqualizer(simulator.rg, stream_management)
        else:
            raise NotImplementedError("BER evaluation supports SISO and the current sionna-backed 2x2 MIMO path only.")
    elif not _supports_siso_ber(config):
        raise NotImplementedError("BER evaluation currently supports SISO for non-Sionna backends.")

    output_dir = config.evaluation_root
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ber_results.csv"
    fig_path = output_dir / "ber_curve.png"

    rows: list[dict[str, float]] = []
    for snr_db in config.evaluation.snr_db:
        metrics = {"ls_lin": 0.0, "lmmse": 0.0, "dl": 0.0}
        for _ in range(config.evaluation.batches_per_snr):
            batch = simulator.generate_batch(config.evaluation.batch_size, snr_db=snr_db)
            with torch.no_grad():
                h_dl = model.predict_channel(batch.h_ls_lin.to(device)).cpu()

            if config.experiment.backend == "sionna" and batch.tx_bits_data is not None and _supports_siso_ber(config):
                equalized_ls = _one_tap_lmmse_equalize(batch.y_rg, batch.h_ls_lin, batch.noise_var)
                equalized_lmmse = _one_tap_lmmse_equalize(batch.y_rg, batch.h_lmmse, batch.noise_var)
                equalized_dl = _one_tap_lmmse_equalize(batch.y_rg, h_dl, batch.noise_var)
                metrics["ls_lin"] += _ber_for_estimate_sionna(equalized_ls, batch.tx_bits_data, batch.noise_var, resource_grid_demapper, bit_demapper)
                metrics["lmmse"] += _ber_for_estimate_sionna(equalized_lmmse, batch.tx_bits_data, batch.noise_var, resource_grid_demapper, bit_demapper)
                metrics["dl"] += _ber_for_estimate_sionna(equalized_dl, batch.tx_bits_data, batch.noise_var, resource_grid_demapper, bit_demapper)
            elif config.experiment.backend == "sionna" and batch.tx_bits_data is not None and _supports_mimo_ber(config):
                metrics["ls_lin"] += _ber_for_estimate_sionna_mimo(simulator, batch.h_ls_lin, batch.y_rg, batch.tx_bits_data, batch.noise_var, mimo_equalizer, bit_demapper)
                metrics["lmmse"] += _ber_for_estimate_sionna_mimo(simulator, batch.h_lmmse, batch.y_rg, batch.tx_bits_data, batch.noise_var, mimo_equalizer, bit_demapper)
                metrics["dl"] += _ber_for_estimate_sionna_mimo(simulator, h_dl, batch.y_rg, batch.tx_bits_data, batch.noise_var, mimo_equalizer, bit_demapper)
            else:
                metrics["ls_lin"] += _ber_for_estimate(batch.y_rg, batch.h_ls_lin, batch.tx_bits, batch.data_mask, batch.noise_var)
                metrics["lmmse"] += _ber_for_estimate(batch.y_rg, batch.h_lmmse, batch.tx_bits, batch.data_mask, batch.noise_var)
                metrics["dl"] += _ber_for_estimate(batch.y_rg, h_dl, batch.tx_bits, batch.data_mask, batch.noise_var)

        rows.append(
            {
                "snr_db": float(snr_db),
                "ls_lin": metrics["ls_lin"] / config.evaluation.batches_per_snr,
                "lmmse": metrics["lmmse"] / config.evaluation.batches_per_snr,
                "dl": metrics["dl"] / config.evaluation.batches_per_snr,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    snrs = [item["snr_db"] for item in rows]
    plt.figure(figsize=(8, 5))
    for name, label in (("ls_lin", "LS-lin"), ("lmmse", "LMMSE-like"), ("dl", "DL")):
        plt.semilogy(snrs, [item[name] for item in rows], marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return fig_path
