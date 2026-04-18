from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class ChannelEstimatorCNN(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 2, base_channels: int = 64, num_res_blocks: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResBlock(base_channels) for _ in range(num_res_blocks)))
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        return self.head(self.body(features))

    def predict_channel(self, ls_estimate: torch.Tensor) -> torch.Tensor:
        return ls_estimate + self(ls_estimate)


class MIMOFactorizedEstimator(nn.Module):
    """MIMO-aware estimator with per-link processing and cross-link fusion.

    The input tensor is interpreted as `num_complex_links` complex-valued links,
    each represented by two real-valued channels (real/imag). Each link is first
    encoded independently with shared weights, then all links are fused jointly
    through cross-link 1x1 mixing, and finally each link is refined and decoded
    back to a residual estimate.
    """

    def __init__(
        self,
        *,
        num_complex_links: int,
        in_channels: int,
        out_channels: int,
        link_channels: int = 24,
        mix_channels: int = 96,
        per_link_res_blocks: int = 2,
        cross_res_blocks: int = 4,
        refine_res_blocks: int = 1,
    ):
        super().__init__()
        expected_channels = 2 * num_complex_links
        if in_channels != expected_channels or out_channels != expected_channels:
            raise ValueError(
                f"MIMOFactorizedEstimator expects in/out channels = 2*num_complex_links = {expected_channels}, "
                f"got in={in_channels}, out={out_channels}."
            )

        self.num_complex_links = num_complex_links
        self.link_channels = link_channels

        self.link_stem = nn.Sequential(
            nn.Conv2d(2, link_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.link_body = nn.Sequential(*(ResBlock(link_channels) for _ in range(per_link_res_blocks)))

        self.link_embedding = nn.Parameter(torch.zeros(1, num_complex_links, link_channels, 1, 1))

        fused_channels = num_complex_links * link_channels
        self.cross_stem = nn.Sequential(
            nn.Conv2d(fused_channels, mix_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.cross_body = nn.Sequential(*(ResBlock(mix_channels) for _ in range(cross_res_blocks)))
        self.cross_head = nn.Conv2d(mix_channels, fused_channels, kernel_size=1, bias=True)

        self.link_refine = nn.Sequential(*(ResBlock(link_channels) for _ in range(refine_res_blocks)))
        self.link_head = nn.Conv2d(link_channels, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_symbols, num_subcarriers = inputs.shape
        expected_channels = 2 * self.num_complex_links
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {channels}.")

        # [B, 2L, S, F] -> [B, L, 2, S, F]
        per_link = inputs.reshape(batch_size, self.num_complex_links, 2, num_symbols, num_subcarriers)

        # Shared per-link encoder
        local = per_link.reshape(batch_size * self.num_complex_links, 2, num_symbols, num_subcarriers)
        local = self.link_stem(local)
        local = self.link_body(local)
        local = local.reshape(batch_size, self.num_complex_links, self.link_channels, num_symbols, num_subcarriers)
        local = local + self.link_embedding

        # Cross-link fusion
        fused = local.reshape(batch_size, self.num_complex_links * self.link_channels, num_symbols, num_subcarriers)
        fused = self.cross_stem(fused)
        fused = self.cross_body(fused)
        fused = self.cross_head(fused)
        fused = fused.reshape(batch_size, self.num_complex_links, self.link_channels, num_symbols, num_subcarriers)

        # Per-link refinement after global mixing
        refined = (local + fused).reshape(batch_size * self.num_complex_links, self.link_channels, num_symbols, num_subcarriers)
        refined = self.link_refine(refined)
        outputs = self.link_head(refined)
        return outputs.reshape(batch_size, expected_channels, num_symbols, num_subcarriers)

    def predict_channel(self, ls_estimate: torch.Tensor) -> torch.Tensor:
        return ls_estimate + self(ls_estimate)
