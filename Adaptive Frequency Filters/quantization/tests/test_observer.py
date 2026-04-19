"""Tests for quantization/observer.py."""

import pytest
import torch

from quantization.observer import (
    ASYMMETRIC,
    CALIBRATING,
    DISABLED,
    FROZEN,
    OBSERVER_REGISTRY,
    SYMMETRIC,
    BaseObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    PercentileObserver,
    build_observer,
    register_observer,
)


torch.manual_seed(0)


# -------- registry -------- #


def test_min_max_is_registered():
    assert "min_max" in OBSERVER_REGISTRY
    assert OBSERVER_REGISTRY["min_max"] is MinMaxObserver


def test_build_observer_constructs_class():
    obs = build_observer("min_max", bits=8, scheme=SYMMETRIC)
    assert isinstance(obs, MinMaxObserver)
    assert obs.bits == 8
    assert obs.scheme == SYMMETRIC


def test_build_observer_unknown_name_raises():
    with pytest.raises(KeyError):
        build_observer("does_not_exist", bits=8, scheme=SYMMETRIC)


def test_register_duplicate_raises():
    with pytest.raises(ValueError):
        @register_observer("min_max")
        class _Duplicate(BaseObserver):
            pass


# -------- state machine -------- #


def test_default_mode_is_disabled():
    obs = MinMaxObserver(bits=8, scheme=SYMMETRIC)
    assert int(obs.mode.item()) == DISABLED


def test_disabled_is_passthrough():
    obs = MinMaxObserver(bits=4, scheme=SYMMETRIC)
    x = torch.randn(100)
    y = obs(x)
    assert torch.equal(x, y)
    # stats should not have been updated
    assert obs.min_val.item() == float("inf")


def test_calibrating_collects_and_passes_through():
    obs = MinMaxObserver(bits=8, scheme=SYMMETRIC)
    obs.set_mode(CALIBRATING)
    x = torch.tensor([-3.0, 2.0, 1.0])
    y = obs(x)
    assert torch.equal(x, y)
    assert obs.min_val.item() == -3.0
    assert obs.max_val.item() == 2.0


def test_calibrating_accumulates_across_batches():
    obs = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    obs(torch.tensor([0.5, 1.0]))  # min=0.5, max=1.0
    obs(torch.tensor([-2.0, 0.0]))  # min=-2.0, max=1.0
    obs(torch.tensor([3.0, 0.5]))  # min=-2.0, max=3.0
    assert obs.min_val.item() == -2.0
    assert obs.max_val.item() == 3.0


def test_set_calibrating_resets_stats():
    obs = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    obs(torch.tensor([-5.0, 5.0]))
    obs.set_mode(CALIBRATING)  # re-entering CAL should reset
    assert obs.min_val.item() == float("inf")
    assert obs.max_val.item() == float("-inf")


def test_freeze_without_observe_raises():
    obs = MinMaxObserver(bits=8, scheme=SYMMETRIC)
    with pytest.raises(RuntimeError, match="no data observed"):
        obs.freeze()


def test_frozen_applies_fake_quant():
    obs = MinMaxObserver(bits=4, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    x = torch.linspace(-1.0, 3.0, 100)
    obs(x)
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
    y = obs(x)
    assert not torch.equal(x, y)
    # 4-bit asymmetric → 16 distinct levels
    assert torch.unique(y).numel() == 16


# -------- scheme correctness -------- #


def test_symmetric_freeze_sets_zp_zero():
    obs = MinMaxObserver(bits=8, scheme=SYMMETRIC)
    obs.set_mode(CALIBRATING)
    obs(torch.tensor([-2.0, 3.0]))  # asymmetric input
    obs.freeze()
    assert obs.zero_point.item() == 0


def test_asymmetric_freeze_produces_valid_zp():
    obs = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    obs(torch.tensor([-1.0, 5.0]))
    obs.freeze()
    zp = int(obs.zero_point.item())
    assert obs.qmin <= zp <= obs.qmax


def test_symmetric_uses_max_abs_not_signed_max():
    obs = MinMaxObserver(bits=8, scheme=SYMMETRIC)
    obs.set_mode(CALIBRATING)
    # min is more extreme than max; symmetric scale must use max(|min|, |max|)
    obs(torch.tensor([-10.0, 3.0]))
    obs.freeze()
    expected_scale = 10.0 / obs.qmax
    assert abs(obs.scale.item() - expected_scale) < 1e-6


# -------- weight use case (observe-then-freeze pattern) -------- #


def test_weight_observe_then_freeze_workflow():
    """Canonical weight use: observe once, freeze, then forward is fake-quant."""
    obs = MinMaxObserver(bits=4, scheme=SYMMETRIC)
    w = torch.randn(16, 3, 3, 3) * 0.1
    obs.observe(w)
    obs.freeze()
    w_q = obs(w)
    assert w_q.shape == w.shape
    # 4-bit symmetric → 15 distinct values max
    assert torch.unique(w_q).numel() <= 15


# -------- device / state_dict ---------- #


# -------- percentile observer -------- #


def test_percentile_registered():
    assert "percentile" in OBSERVER_REGISTRY
    assert OBSERVER_REGISTRY["percentile"] is PercentileObserver


def test_percentile_ignores_outliers():
    """A single 10-sigma outlier should NOT balloon the scale."""
    torch.manual_seed(0)
    x = torch.randn(10000)  # ~N(0,1), typical range ~[-4, 4]
    x[0] = 1000.0  # outlier

    p_obs = PercentileObserver(bits=8, scheme=ASYMMETRIC, low_percentile=0.001, high_percentile=0.999)
    p_obs.set_mode(CALIBRATING)
    p_obs(x)
    p_obs.freeze()

    mm_obs = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    mm_obs.set_mode(CALIBRATING)
    mm_obs(x)
    mm_obs.freeze()

    # Percentile scale should be ~1000x smaller than min/max scale (because min/max sees 1000)
    assert p_obs.scale.item() < mm_obs.scale.item() / 50
    # Typical values round-trip accurately with percentile
    typical = torch.randn(100)
    err_p = (typical - p_obs(typical)).abs().mean().item()
    err_m = (typical - mm_obs(typical)).abs().mean().item()
    assert err_p < err_m / 10, f"percentile err={err_p} should be << min/max err={err_m}"


def test_percentile_freeze_without_observe_raises():
    obs = PercentileObserver(bits=8, scheme=ASYMMETRIC)
    with pytest.raises(RuntimeError, match="n_batches=0"):
        obs.freeze()


def test_percentile_accumulates_across_batches():
    obs = PercentileObserver(bits=8, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    for _ in range(5):
        obs(torch.randn(1000))
    assert obs.n_batches.item() == 5
    obs.freeze()
    assert torch.isfinite(obs.scale)


def test_percentile_build_via_registry():
    obs = build_observer("percentile", bits=8, scheme=ASYMMETRIC)
    assert isinstance(obs, PercentileObserver)


# -------- per-channel observer (weights) -------- #


def test_per_channel_registered():
    assert "per_channel_min_max" in OBSERVER_REGISTRY
    assert OBSERVER_REGISTRY["per_channel_min_max"] is PerChannelMinMaxObserver


def test_per_channel_scale_has_per_out_channel_shape():
    """For a Conv2d weight (out_c, in_c, kH, kW) the scale should be shape (out_c,)."""
    obs = PerChannelMinMaxObserver(bits=8, scheme=SYMMETRIC, axis=0)
    w = torch.randn(16, 3, 3, 3)
    obs.observe(w)
    obs.freeze()
    assert obs.scale.shape == (16,)


def test_per_channel_gives_each_channel_its_own_range():
    """Channel magnitudes differ by 100x — per-channel must not crush small channels."""
    obs = PerChannelMinMaxObserver(bits=8, scheme=SYMMETRIC, axis=0)
    w = torch.zeros(4, 3, 3, 3)
    w[0] = 0.01
    w[1] = 1.0
    w[2] = -50.0  # giant channel
    w[3] = 0.02
    obs.observe(w)
    obs.freeze()
    # scale for channel 0 must be much smaller than channel 2
    assert obs.scale[0].item() < obs.scale[2].item() / 100


def test_per_channel_fake_quantize_broadcasts_correctly():
    """fake_quantize must apply per-channel scale along axis=0 on a weight tensor."""
    obs = PerChannelMinMaxObserver(bits=4, scheme=SYMMETRIC, axis=0)
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3)
    # Scale up some channels dramatically so per-channel matters
    w[2] *= 100.0
    w[5] *= 0.001
    obs.observe(w)
    obs.freeze()
    w_q = obs(w)
    assert w_q.shape == w.shape
    # Small channel 5 should still have nonzero (non-collapsed) values
    assert torch.unique(w_q[5]).numel() >= 3


def test_per_channel_symmetric_zero_stays_zero():
    obs = PerChannelMinMaxObserver(bits=4, scheme=SYMMETRIC, axis=0)
    w = torch.zeros(3, 2, 3, 3)
    obs.observe(w)
    obs.freeze()
    w_q = obs(w)
    assert torch.all(w_q == 0.0)


def test_per_channel_reduces_error_vs_per_tensor_on_depthwise():
    """The whole point: per-channel beats per-tensor when channel magnitudes vary."""
    torch.manual_seed(0)
    w = torch.randn(16, 1, 3, 3)  # depthwise shape (out=16, in=1, ...)
    # Introduce magnitude imbalance
    w[0] *= 50.0
    w[1] *= 0.01

    per_tensor = MinMaxObserver(bits=4, scheme=SYMMETRIC)
    per_tensor.observe(w)
    per_tensor.freeze()
    err_pt = (w - per_tensor(w)).abs().mean().item()

    per_ch = PerChannelMinMaxObserver(bits=4, scheme=SYMMETRIC, axis=0)
    per_ch.observe(w)
    per_ch.freeze()
    err_pc = (w - per_ch(w)).abs().mean().item()

    assert err_pc < err_pt / 2, f"per-channel ({err_pc}) should be much better than per-tensor ({err_pt})"


def test_per_channel_build_via_registry():
    obs = build_observer("per_channel_min_max", bits=8, scheme=SYMMETRIC)
    assert isinstance(obs, PerChannelMinMaxObserver)


# -------- buffer roundtrip -------- #


def test_buffers_move_with_module():
    obs = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    obs.set_mode(CALIBRATING)
    obs(torch.tensor([0.0, 2.0]))
    obs.freeze()
    sd = obs.state_dict()
    assert "scale" in sd
    assert "zero_point" in sd
    assert "min_val" in sd
    assert "max_val" in sd
    assert "mode" in sd

    # round-trip via state_dict
    obs2 = MinMaxObserver(bits=8, scheme=ASYMMETRIC)
    obs2.load_state_dict(sd)
    assert int(obs2.mode.item()) == FROZEN
    x = torch.tensor([0.5, 1.5])
    assert torch.equal(obs(x), obs2(x))
