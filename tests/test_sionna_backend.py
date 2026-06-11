import importlib.util

import pytest

from ofdm_ce.config import ProjectConfig
from ofdm_ce.sionna_backend import create_simulator


pytestmark = pytest.mark.skipif(importlib.util.find_spec("sionna") is None, reason="Sionna is not installed")


def test_sionna_mimo_2x2_cdl_c_batch_shapes() -> None:
    config = ProjectConfig.from_mapping(
        {
            "experiment": {"name": "mimo_2x2_cdl_c_test", "backend": "sionna", "device": "cpu"},
            "radio": {
                "num_subcarriers": 64,
                "num_ofdm_symbols": 14,
                "pilot_ofdm_symbol_indices": [2, 11],
                "pilot_pattern": "kronecker",
                "num_tx": 1,
                "num_streams_per_tx": 2,
                "num_tx_ant": 2,
                "num_rx_ant": 2,
                "num_bits_per_symbol": 2,
                "channel_model": "CDL-C",
                "delay_spread_s": 1e-7,
                "speed_m_per_s": 10.0,
                "carrier_frequency_hz": 3.5e9,
                "subcarrier_spacing_hz": 15e3,
            },
        }
    )

    simulator = create_simulator(config)
    batch = simulator.generate_batch(batch_size=2, snr_db=5.0)

    assert batch.h_true.shape == (2, 8, 14, 64)
    assert batch.h_ls_lin.shape == batch.h_true.shape
    assert batch.h_lmmse.shape == batch.h_true.shape
    assert batch.y_rg.shape == (2, 4, 14, 64)
    assert batch.x_rg.shape == (2, 4, 14, 64)
    assert batch.tx_bits_data is not None
    assert batch.tx_bits_data.shape[0] == 2
    assert batch.tx_bits_data.shape[-1] == 2

    h_true = simulator.unflatten_h(batch.h_true)
    y_rg = simulator.unflatten_y(batch.y_rg)

    assert h_true.shape == (2, 1, 2, 1, 2, 14, 64)
    assert y_rg.shape == (2, 1, 2, 14, 64)
