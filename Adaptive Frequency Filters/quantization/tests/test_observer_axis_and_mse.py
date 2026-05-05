"""Sanity tests for the Phase 9 observer changes:
  - per-channel observers accept axis=1 (Conv NCHW act) and axis=-1 (Linear act)
  - MSE / per_channel_mse work in both schemes and across multiple observe() calls
"""
from __future__ import annotations

import torch
import pytest

from quantization.observer import (
    OBSERVER_REGISTRY,
    CALIBRATING,
    FROZEN,
    build_observer,
)

PER_CHANNEL = ("per_channel_min_max", "per_channel_percentile",
               "per_channel_mse", "per_channel_histogram")
PER_TENSOR = ("min_max", "percentile", "mse", "histogram")
ALL = PER_CHANNEL + PER_TENSOR


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("scheme", ["symmetric", "asymmetric"])
def test_observer_works_on_conv_act_shape(name, scheme):
    """Build observer with axis=1 (Conv NCHW), observe across two batches, freeze, forward."""
    obs = build_observer(name, bits=8, scheme=scheme, axis=1)
    obs.set_mode(CALIBRATING)
    torch.manual_seed(0)
    for _ in range(2):
        x = torch.randn(4, 16, 8, 8)
        out = obs(x)
        assert out.shape == x.shape
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
    y = obs(torch.randn(4, 16, 8, 8))
    assert y.shape == (4, 16, 8, 8)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("scheme", ["symmetric", "asymmetric"])
def test_observer_works_on_linear_act_shape(name, scheme):
    """axis=-1 should work for Linear inputs (B, F)."""
    obs = build_observer(name, bits=8, scheme=scheme, axis=-1)
    obs.set_mode(CALIBRATING)
    torch.manual_seed(1)
    for _ in range(2):
        x = torch.randn(8, 64)
        obs(x)
    obs.freeze()
    y = obs(torch.randn(8, 64))
    assert y.shape == (8, 64)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("scheme", ["symmetric", "asymmetric"])
def test_observer_works_on_weight_shape(name, scheme):
    """axis=0 for Conv2d weight (out_c, in_c, kH, kW)."""
    obs = build_observer(name, bits=8, scheme=scheme, axis=0)
    w = torch.randn(32, 16, 3, 3)
    obs.set_mode(CALIBRATING)
    obs(w)
    obs.freeze()
    y = obs(w)
    assert y.shape == w.shape


def test_mse_asymmetric_scale_is_reasonable():
    """Asymmetric MSE on a heavy-tailed distribution should not collapse."""
    obs = build_observer("mse", bits=8, scheme="asymmetric")
    obs.set_mode(CALIBRATING)
    torch.manual_seed(2)
    # Mostly bounded, with rare outliers
    base = torch.randn(10000) * 0.3
    base[:5] = 50.0
    obs(base)
    obs(base[5:])
    obs.freeze()
    s = obs.scale.item()
    assert 0 < s < 1.0, f"asym MSE scale unexpectedly large: {s}"


def test_per_channel_mse_per_channel_axis():
    """per_channel_mse on axis=1 should produce one scale per input channel."""
    obs = build_observer("per_channel_mse", bits=8, scheme="asymmetric", axis=1)
    obs.set_mode(CALIBRATING)
    torch.manual_seed(3)
    for _ in range(3):
        x = torch.randn(4, 12, 8, 8)
        obs(x)
    obs.freeze()
    assert obs.scale.numel() == 12
    assert obs.zero_point.numel() == 12
