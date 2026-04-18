import torch

from ofdm_ce.model import ChannelEstimatorCNN, MIMOFactorizedEstimator


def test_model_forward_shape() -> None:
    model = ChannelEstimatorCNN(in_channels=2, base_channels=16, num_res_blocks=2)
    inputs = torch.randn(3, 2, 14, 256)
    outputs = model(inputs)
    assert outputs.shape == inputs.shape


def test_model_backward() -> None:
    model = ChannelEstimatorCNN(in_channels=2, base_channels=16, num_res_blocks=2)
    inputs = torch.randn(3, 2, 14, 256)
    target = torch.randn(3, 2, 14, 256)
    loss = (model(inputs) - target).pow(2).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(grad is not None for grad in gradients)


def test_mimo_factorized_forward_shape() -> None:
    model = MIMOFactorizedEstimator(
        num_complex_links=4,
        in_channels=8,
        out_channels=8,
        link_channels=8,
        mix_channels=16,
        per_link_res_blocks=1,
        cross_res_blocks=1,
        refine_res_blocks=1,
    )
    inputs = torch.randn(2, 8, 14, 64)
    outputs = model(inputs)
    assert outputs.shape == inputs.shape


def test_mimo_factorized_backward() -> None:
    model = MIMOFactorizedEstimator(
        num_complex_links=4,
        in_channels=8,
        out_channels=8,
        link_channels=8,
        mix_channels=16,
        per_link_res_blocks=1,
        cross_res_blocks=1,
        refine_res_blocks=1,
    )
    inputs = torch.randn(2, 8, 14, 64)
    target = torch.randn(2, 8, 14, 64)
    loss = (model(inputs) - target).pow(2).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(grad is not None for grad in gradients)
