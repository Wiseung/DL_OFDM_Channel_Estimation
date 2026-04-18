from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .complex_utils import channels_to_complex_tensor, complex_tensor_to_channels
from .config import ProjectConfig


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional dependency is missing."""


@dataclass(slots=True)
class SimulatorBatch:
    h_true: torch.Tensor
    h_ls_nn: torch.Tensor
    h_ls_lin: torch.Tensor
    h_ls_lin_time_avg: torch.Tensor
    h_lmmse: torch.Tensor
    y_rg: torch.Tensor
    x_rg: torch.Tensor
    pilot_mask: torch.Tensor
    data_mask: torch.Tensor
    snr_db: torch.Tensor
    noise_var: torch.Tensor
    tx_bits: torch.Tensor
    tx_bits_data: torch.Tensor | None = None


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def create_simulator(config: ProjectConfig):
    if config.experiment.backend == "synthetic":
        return SyntheticOFDMSimulator(config)
    if config.experiment.backend == "sionna":
        return SionnaOFDMSimulator(config)
    raise ValueError(f"Unsupported backend: {config.experiment.backend}")


def qpsk_map(bits: torch.Tensor) -> torch.Tensor:
    if bits.shape[-1] != 2:
        raise ValueError("Expected 2 bits per symbol for QPSK.")
    b0 = bits[..., 0].float()
    b1 = bits[..., 1].float()
    real = 1.0 - 2.0 * b1
    imag = 1.0 - 2.0 * b0
    return torch.complex(real, imag) / (2.0 ** 0.5)


def qpsk_demapper(symbols: torch.Tensor) -> torch.Tensor:
    bits = torch.zeros(*symbols.shape, 2, device=symbols.device)
    bits[..., 0] = (symbols.imag < 0).float()
    bits[..., 1] = (symbols.real < 0).float()
    return bits


def lmmse_like_smoothing(ls_lin: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 8.0, 2.0], [1.0, 2.0, 1.0]],
        device=ls_lin.device,
        dtype=ls_lin.dtype,
    )
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, 3, 3).repeat(ls_lin.shape[1], 1, 1, 1)
    return F.conv2d(ls_lin, weight, padding=1, groups=ls_lin.shape[1])


class BaseOFDMSimulator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = resolve_device(config.experiment.device)
        self.num_rx = 1
        self.num_symbols = config.radio.num_ofdm_symbols
        self.num_subcarriers = config.radio.num_subcarriers
        self.pilot_indices = tuple(config.radio.pilot_ofdm_symbol_indices)
        self.pilot_mask = torch.zeros(self.num_symbols, self.num_subcarriers, dtype=torch.bool, device=self.device)
        self.pilot_mask[list(self.pilot_indices), :] = True
        self.data_mask = ~self.pilot_mask

    def sample_snr(self, batch_size: int, snr_db: float | torch.Tensor | None = None) -> torch.Tensor:
        if snr_db is None:
            low = self.config.dataset.train_snr_min_db
            high = self.config.dataset.train_snr_max_db
            return torch.empty(batch_size, device=self.device).uniform_(low, high)
        if isinstance(snr_db, torch.Tensor):
            return snr_db.to(self.device)
        return torch.full((batch_size,), float(snr_db), device=self.device)

    def generate_batch(self, batch_size: int, snr_db: float | torch.Tensor | None = None) -> SimulatorBatch:
        raise NotImplementedError

    def _flatten_complex_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(tensor)
        if tensor.ndim < 3 or not torch.is_complex(tensor):
            raise ValueError(f"Expected complex tensor with shape [B, ..., S, F], got {tensor.shape}")
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[-2], tensor.shape[-1])
        return torch.cat([tensor.real, tensor.imag], dim=1)


class SyntheticOFDMSimulator(BaseOFDMSimulator):
    def _smooth_complex(self, complex_grid: torch.Tensor, passes: int = 3) -> torch.Tensor:
        channels = complex_tensor_to_channels(complex_grid)
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=channels.device,
            dtype=channels.dtype,
        )
        kernel = kernel / kernel.sum()
        weight = kernel.view(1, 1, 3, 3).repeat(channels.shape[1], 1, 1, 1)
        for _ in range(passes):
            channels = F.conv2d(channels, weight, padding=1, groups=channels.shape[1])
        return channels_to_complex_tensor(channels)

    def _build_channel(self, batch_size: int) -> torch.Tensor:
        base = torch.complex(
            torch.randn(batch_size, self.num_symbols, self.num_subcarriers, device=self.device),
            torch.randn(batch_size, self.num_symbols, self.num_subcarriers, device=self.device),
        )
        channel = self._smooth_complex(base)
        power = channel.abs().pow(2).mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6).sqrt()
        return channel / power

    def _build_transmit_grid(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_rg = torch.zeros(batch_size, self.num_symbols, self.num_subcarriers, dtype=torch.cfloat, device=self.device)
        tx_bits = torch.zeros(batch_size, self.num_symbols, self.num_subcarriers, 2, device=self.device)

        x_rg.view(batch_size, -1)[:, self.pilot_mask.view(-1)] = 1.0 + 0.0j

        num_data = int(self.data_mask.sum().item())
        data_bits = torch.randint(0, 2, (batch_size, num_data, 2), device=self.device)
        data_symbols = qpsk_map(data_bits)
        x_rg.view(batch_size, -1)[:, self.data_mask.view(-1)] = data_symbols
        tx_bits.view(batch_size, -1, 2)[:, self.data_mask.view(-1), :] = data_bits.float()
        return x_rg, tx_bits, data_bits

    def _interpolate_pilots(self, pilot_estimates: torch.Tensor, mode: str) -> torch.Tensor:
        pilot_values = pilot_estimates[:, list(self.pilot_indices), :]
        output = torch.zeros_like(pilot_estimates)

        if mode == "nn":
            for symbol_idx in range(self.num_symbols):
                nearest = min(self.pilot_indices, key=lambda pilot_idx: abs(pilot_idx - symbol_idx))
                source_idx = self.pilot_indices.index(nearest)
                output[:, symbol_idx, :] = pilot_values[:, source_idx, :]
            return output

        if mode == "lin":
            start, end = self.pilot_indices[0], self.pilot_indices[-1]
            start_values = pilot_values[:, 0, :]
            end_values = pilot_values[:, -1, :]
            for symbol_idx in range(self.num_symbols):
                if symbol_idx <= start:
                    output[:, symbol_idx, :] = start_values
                elif symbol_idx >= end:
                    output[:, symbol_idx, :] = end_values
                else:
                    alpha = (symbol_idx - start) / max(end - start, 1)
                    output[:, symbol_idx, :] = (1.0 - alpha) * start_values + alpha * end_values
            return output

        if mode == "lin_time_avg":
            averaged = pilot_values.mean(dim=1, keepdim=True)
            output[:] = averaged
            return output

        raise ValueError(f"Unknown interpolation mode: {mode}")

    def generate_batch(self, batch_size: int, snr_db: float | torch.Tensor | None = None) -> SimulatorBatch:
        if (
            self.config.radio.num_tx != 1
            or self.config.radio.num_streams_per_tx != 1
            or self.config.radio.num_tx_ant != 1
            or self.config.radio.num_rx_ant != 1
        ):
            raise NotImplementedError("The synthetic backend currently supports SISO only.")

        sampled_snr_db = self.sample_snr(batch_size, snr_db)
        noise_var = torch.pow(10.0, -sampled_snr_db / 10.0)

        h_true_complex = self._build_channel(batch_size)
        x_rg_complex, tx_bits, data_bits = self._build_transmit_grid(batch_size)
        noise = torch.complex(
            torch.randn_like(h_true_complex.real),
            torch.randn_like(h_true_complex.imag),
        )
        noise = noise * (noise_var.view(-1, 1, 1) / 2.0).sqrt()
        y_rg_complex = h_true_complex * x_rg_complex + noise

        pilot_est = torch.zeros_like(h_true_complex)
        pilot_mask_flat = self.pilot_mask.view(-1)
        pilot_est.view(batch_size, -1)[:, pilot_mask_flat] = (
            y_rg_complex.view(batch_size, -1)[:, pilot_mask_flat]
            / x_rg_complex.view(batch_size, -1)[:, pilot_mask_flat]
        )

        h_ls_nn = self._interpolate_pilots(pilot_est, mode="nn")
        h_ls_lin = self._interpolate_pilots(pilot_est, mode="lin")
        h_ls_lin_time_avg = self._interpolate_pilots(pilot_est, mode="lin_time_avg")
        h_lmmse = channels_to_complex_tensor(lmmse_like_smoothing(complex_tensor_to_channels(h_ls_lin)))

        return SimulatorBatch(
            h_true=complex_tensor_to_channels(h_true_complex),
            h_ls_nn=complex_tensor_to_channels(h_ls_nn),
            h_ls_lin=complex_tensor_to_channels(h_ls_lin),
            h_ls_lin_time_avg=complex_tensor_to_channels(h_ls_lin_time_avg),
            h_lmmse=complex_tensor_to_channels(h_lmmse),
            y_rg=complex_tensor_to_channels(y_rg_complex),
            x_rg=complex_tensor_to_channels(x_rg_complex),
            pilot_mask=self.pilot_mask.detach().cpu(),
            data_mask=self.data_mask.detach().cpu(),
            snr_db=sampled_snr_db.detach().cpu(),
            noise_var=noise_var.detach().cpu(),
            tx_bits=tx_bits.detach().cpu(),
            tx_bits_data=data_bits.float().detach().cpu(),
        )


class SionnaOFDMSimulator(BaseOFDMSimulator):
    """Best-effort adapter for the official Sionna OFDM stack."""

    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        try:
            import sionna  # type: ignore
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Sionna is not installed. Run scripts/setup_wsl.sh or use backend=synthetic."
            ) from exc

        self.sionna = sionna
        self._build_modules()

    def _import_attr(self, module_paths: tuple[str, ...], attr_name: str) -> Any:
        for module_path in module_paths:
            try:
                module = __import__(module_path, fromlist=[attr_name])
                return getattr(module, attr_name)
            except (ImportError, AttributeError):
                continue
        raise DependencyUnavailableError(f"Unable to import {attr_name} from {module_paths}")

    def _build_modules(self) -> None:
        ResourceGrid = self._import_attr(("sionna.phy.ofdm", "sionna.ofdm"), "ResourceGrid")
        ResourceGridMapper = self._import_attr(("sionna.phy.ofdm", "sionna.ofdm"), "ResourceGridMapper")
        OFDMChannel = self._import_attr(("sionna.phy.channel", "sionna.channel"), "OFDMChannel")
        TDL = self._import_attr(("sionna.phy.channel.tr38901", "sionna.channel.tr38901"), "TDL")
        CDL = self._import_attr(("sionna.phy.channel.tr38901", "sionna.channel.tr38901"), "CDL")
        Antenna = self._import_attr(("sionna.phy.channel.tr38901", "sionna.channel.tr38901"), "Antenna")
        PanelArray = self._import_attr(("sionna.phy.channel.tr38901", "sionna.channel.tr38901"), "PanelArray")
        LSChannelEstimator = self._import_attr(("sionna.phy.ofdm", "sionna.ofdm"), "LSChannelEstimator")
        Mapper = self._import_attr(("sionna.phy.mapping", "sionna.mapping"), "Mapper")
        BinarySource = self._import_attr(("sionna.phy.mapping", "sionna.mapping"), "BinarySource")

        try:
            self.rg = ResourceGrid(
                num_ofdm_symbols=self.num_symbols,
                fft_size=self.num_subcarriers,
                subcarrier_spacing=self.config.radio.subcarrier_spacing_hz,
                num_tx=self.config.radio.num_tx,
                num_streams_per_tx=self.config.radio.num_streams_per_tx,
                pilot_pattern=self.config.radio.pilot_pattern,
                pilot_ofdm_symbol_indices=list(self.pilot_indices),
                num_guard_carriers=(0, 0),
                dc_null=False,
            )
        except TypeError:
            self.rg = ResourceGrid(
                num_ofdm_symbols=self.num_symbols,
                num_subcarriers=self.num_subcarriers,
                num_tx=self.config.radio.num_tx,
                num_streams_per_tx=self.config.radio.num_streams_per_tx,
                pilot_pattern=self.config.radio.pilot_pattern,
                pilot_ofdm_symbol_indices=list(self.pilot_indices),
            )

        self.rg_mapper = ResourceGridMapper(self.rg)
        self.bit_source = BinarySource()
        try:
            self.mapper = Mapper("qam", self.config.radio.num_bits_per_symbol)
        except TypeError:
            self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=self.config.radio.num_bits_per_symbol)

        channel_model_name = self.config.radio.channel_model.upper()
        model_letter = self.config.radio.channel_model.split("-")[-1]
        if channel_model_name.startswith("TDL"):
            try:
                channel_model = TDL(
                    model=model_letter,
                    delay_spread=self.config.radio.delay_spread_s,
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                    min_speed=self.config.radio.speed_m_per_s,
                    max_speed=self.config.radio.speed_m_per_s,
                    num_rx_ant=self.config.radio.num_rx_ant,
                    num_tx_ant=self.config.radio.num_tx_ant,
                )
            except TypeError:
                channel_model = TDL(
                    model_letter,
                    delay_spread=self.config.radio.delay_spread_s,
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                    min_speed=self.config.radio.speed_m_per_s,
                    max_speed=self.config.radio.speed_m_per_s,
                    num_rx_ant=self.config.radio.num_rx_ant,
                    num_tx_ant=self.config.radio.num_tx_ant,
                )
        elif channel_model_name.startswith("CDL"):
            if self.config.radio.num_rx_ant == 1:
                ut_array = Antenna(
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                )
            else:
                ut_array = PanelArray(
                    num_rows_per_panel=1,
                    num_cols_per_panel=self.config.radio.num_rx_ant,
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                )
            if self.config.radio.num_tx_ant == 1:
                bs_array = Antenna(
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                )
            else:
                bs_array = PanelArray(
                    num_rows_per_panel=1,
                    num_cols_per_panel=self.config.radio.num_tx_ant,
                    polarization="single",
                    polarization_type="V",
                    antenna_pattern="omni",
                    carrier_frequency=self.config.radio.carrier_frequency_hz,
                )
            channel_model = CDL(
                model=model_letter,
                delay_spread=self.config.radio.delay_spread_s,
                carrier_frequency=self.config.radio.carrier_frequency_hz,
                ut_array=ut_array,
                bs_array=bs_array,
                min_speed=self.config.radio.speed_m_per_s,
                max_speed=self.config.radio.speed_m_per_s,
            )
        else:
            raise ValueError(f"Unsupported channel model: {self.config.radio.channel_model}")

        self.channel = OFDMChannel(
            channel_model=channel_model,
            resource_grid=self.rg,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True,
        )

        self.ls_nn = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.ls_lin = LSChannelEstimator(self.rg, interpolation_type="lin")
        self.ls_lin_time_avg = LSChannelEstimator(self.rg, interpolation_type="lin_time_avg")

        pilot_mask = getattr(getattr(self.rg, "pilot_pattern", None), "mask", None)
        if pilot_mask is not None:
            pilot_mask = torch.as_tensor(pilot_mask)
            while pilot_mask.ndim > 2 and pilot_mask.shape[0] == 1:
                pilot_mask = pilot_mask.squeeze(0)
            self.pilot_mask = pilot_mask.to(torch.bool).to(self.device)
            self.data_mask = ~self.pilot_mask

    def _call_layer(self, layer: Any, *inputs: Any) -> Any:
        return layer(*inputs)

    def _channels_to_complex(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(tensor)
        num_complex_channels = tensor.shape[1] // 2
        return torch.complex(tensor[:, :num_complex_channels], tensor[:, num_complex_channels:])

    def unflatten_h(self, tensor: torch.Tensor) -> torch.Tensor:
        complex_tensor = self._channels_to_complex(tensor)
        return complex_tensor.reshape(
            complex_tensor.shape[0],
            self.num_rx,
            self.config.radio.num_rx_ant,
            self.config.radio.num_tx,
            self.config.radio.num_streams_per_tx,
            self.num_symbols,
            self.num_subcarriers,
        )

    def unflatten_y(self, tensor: torch.Tensor) -> torch.Tensor:
        complex_tensor = self._channels_to_complex(tensor)
        return complex_tensor.reshape(
            complex_tensor.shape[0],
            self.num_rx,
            self.config.radio.num_rx_ant,
            self.num_symbols,
            self.num_subcarriers,
        )

    def generate_batch(self, batch_size: int, snr_db: float | torch.Tensor | None = None) -> SimulatorBatch:
        sampled_snr_db = self.sample_snr(batch_size, snr_db)
        noise_var = torch.pow(10.0, -sampled_snr_db / 10.0)

        num_data_bits_per_stream = int(self.rg.num_data_symbols) * self.config.radio.num_bits_per_symbol
        bits_shape = [batch_size, self.config.radio.num_tx, self.config.radio.num_streams_per_tx, num_data_bits_per_stream]
        bits = self.bit_source(bits_shape)
        x_symbols = self.mapper(bits)
        x_rg = self._call_layer(self.rg_mapper, x_symbols)
        y_rg, h_freq = self._call_layer(self.channel, x_rg, noise_var)

        h_ls_nn, _ = self._call_layer(self.ls_nn, y_rg, noise_var)
        h_ls_lin, _ = self._call_layer(self.ls_lin, y_rg, noise_var)
        h_ls_lin_time_avg, _ = self._call_layer(self.ls_lin_time_avg, y_rg, noise_var)

        h_true_ri = self._flatten_complex_tensor(torch.as_tensor(h_freq))
        h_ls_nn_ri = self._flatten_complex_tensor(torch.as_tensor(h_ls_nn))
        h_ls_lin_ri = self._flatten_complex_tensor(torch.as_tensor(h_ls_lin))
        h_ls_lin_time_avg_ri = self._flatten_complex_tensor(torch.as_tensor(h_ls_lin_time_avg))
        h_lmmse_ri = lmmse_like_smoothing(h_ls_lin_ri)

        bits_tensor = torch.as_tensor(bits).float().reshape(
            batch_size,
            self.config.radio.num_tx,
            self.config.radio.num_streams_per_tx,
            -1,
            2,
        )
        flat_bits = bits_tensor.reshape(batch_size, -1, 2).detach().cpu()

        return SimulatorBatch(
            h_true=h_true_ri.detach().cpu(),
            h_ls_nn=h_ls_nn_ri.detach().cpu(),
            h_ls_lin=h_ls_lin_ri.detach().cpu(),
            h_ls_lin_time_avg=h_ls_lin_time_avg_ri.detach().cpu(),
            h_lmmse=h_lmmse_ri.detach().cpu(),
            y_rg=self._flatten_complex_tensor(torch.as_tensor(y_rg)).detach().cpu(),
            x_rg=self._flatten_complex_tensor(torch.as_tensor(x_rg)).detach().cpu(),
            pilot_mask=self.pilot_mask.detach().cpu(),
            data_mask=self.data_mask.detach().cpu(),
            snr_db=sampled_snr_db.detach().cpu(),
            noise_var=noise_var.detach().cpu(),
            tx_bits=flat_bits,
            tx_bits_data=flat_bits,
        )
