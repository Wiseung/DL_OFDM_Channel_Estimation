import torch

from ofdm_ce.complex_utils import channels_to_complex_tensor, complex_tensor_to_channels, nmse_linear


def test_complex_round_trip() -> None:
    complex_tensor = torch.complex(torch.randn(4, 14, 256), torch.randn(4, 14, 256))
    channels = complex_tensor_to_channels(complex_tensor)
    restored = channels_to_complex_tensor(channels)
    assert torch.allclose(complex_tensor, restored)


def test_nmse_zero_for_identical_inputs() -> None:
    tensor = torch.randn(8, 2, 14, 256)
    assert torch.isclose(nmse_linear(tensor, tensor), torch.tensor(0.0))

