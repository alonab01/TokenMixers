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
    HistogramObserver,
    MinMaxObserver,
    MSEObserver,
    PerChannelMinMaxObserver,
    PerChannelMSEObserver,
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


# ---------------------------------------------------------------- #
# Phase 6: MSE observers
# ---------------------------------------------------------------- #


def test_mse_registered():
    assert "mse" in OBSERVER_REGISTRY
    assert "per_channel_mse" in OBSERVER_REGISTRY


def test_mse_accepts_both_schemes():
    """Phase 9: MSE observers are now histogram-backed and accept asymmetric too."""
    MSEObserver(bits=8, scheme=ASYMMETRIC)
    PerChannelMSEObserver(bits=8, scheme=ASYMMETRIC)
    MSEObserver(bits=8, scheme=SYMMETRIC)
    PerChannelMSEObserver(bits=8, scheme=SYMMETRIC)


def test_mse_observer_workflow():
    """observe() produces a frozen-ready scale; freeze() flips mode."""
    obs = MSEObserver(bits=4, scheme=SYMMETRIC)
    w = torch.randn(64) * 0.5
    w[0] = 5.0  # outlier
    obs.observe(w)
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
    # forward applies fake-quant
    y = obs(w)
    assert y.shape == w.shape
    # 4-bit symmetric -> at most 15 distinct quantized values
    assert torch.unique(y).numel() <= 15


def test_mse_picks_smaller_scale_than_minmax_with_outlier():
    """The whole point: MSE scale < max(|W|)/qmax when the tensor has an outlier."""
    torch.manual_seed(7)
    w = torch.randn(1000) * 0.3
    w[0] = 10.0  # one outlier, 30 sigma

    mm = MinMaxObserver(bits=4, scheme=SYMMETRIC)
    mm.observe(w)
    mm.freeze()

    mse = MSEObserver(bits=4, scheme=SYMMETRIC)
    mse.observe(w)
    mse.freeze()

    # MSE scale should be strictly smaller than min/max scale.
    assert mse.scale.item() < mm.scale.item()
    # And the resulting quantization error on typical (non-outlier) values
    # should be smaller too.
    typical = w[1:]
    err_mm = (typical - mm(typical)).pow(2).mean().item()
    err_mse = (typical - mse(typical)).pow(2).mean().item()
    assert err_mse < err_mm


def test_per_channel_mse_scale_shape():
    obs = PerChannelMSEObserver(bits=4, scheme=SYMMETRIC, axis=0)
    w = torch.randn(8, 16, 3, 3)
    obs.observe(w)
    obs.freeze()
    assert obs.scale.shape == (8,)


def test_per_channel_mse_beats_per_channel_minmax_on_depthwise_with_outliers():
    """Per-channel MSE should be at least as good as per-channel min/max for depthwise
    weights with channel-wise outliers."""
    torch.manual_seed(11)
    n_out = 16
    w = torch.randn(n_out, 1, 3, 3) * 0.2
    # inject one large outlier per channel
    for c in range(n_out):
        w[c, 0, 0, 0] = 5.0 * (1 + c % 3)  # heterogeneous outliers

    pc_mm = PerChannelMinMaxObserver(bits=4, scheme=SYMMETRIC)
    pc_mm.observe(w)
    pc_mm.freeze()
    err_mm = (w - pc_mm(w)).pow(2).mean().item()

    pc_mse = PerChannelMSEObserver(bits=4, scheme=SYMMETRIC)
    pc_mse.observe(w)
    pc_mse.freeze()
    err_mse = (w - pc_mse(w)).pow(2).mean().item()

    # MSE should be strictly better, or at worst equal (if init scale already optimal).
    assert err_mse <= err_mm * 1.0001, f"MSE worse than min/max: {err_mse} vs {err_mm}"


def test_mse_build_via_registry():
    obs = build_observer("mse", bits=4, scheme=SYMMETRIC)
    assert isinstance(obs, MSEObserver)
    obs.observe(torch.randn(50))
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN

    obs2 = build_observer("per_channel_mse", bits=4, scheme=SYMMETRIC)
    assert isinstance(obs2, PerChannelMSEObserver)
    obs2.observe(torch.randn(8, 16, 3, 3))
    obs2.freeze()
    assert int(obs2.mode.item()) == FROZEN


def test_mse_freeze_without_observe_raises():
    obs = MSEObserver(bits=8, scheme=SYMMETRIC)
    with pytest.raises(RuntimeError, match="no data observed"):
        obs.freeze()
    obs2 = PerChannelMSEObserver(bits=8, scheme=SYMMETRIC)
    with pytest.raises(RuntimeError, match="no data observed"):
        obs2.freeze()


# ---------------------------------------------------------------- #
# Phase 6b: Histogram + KL observer
# ---------------------------------------------------------------- #


def test_histogram_registered():
    assert "histogram" in OBSERVER_REGISTRY


def test_histogram_freeze_without_observe_raises():
    obs = HistogramObserver(bits=8, scheme=ASYMMETRIC, n_bins=256)
    with pytest.raises(RuntimeError, match="no data observed"):
        obs.freeze()


def test_histogram_basic_workflow_asymmetric():
    obs = HistogramObserver(bits=4, scheme=ASYMMETRIC, n_bins=256)
    obs.set_mode(CALIBRATING)
    for _ in range(3):
        obs(torch.randn(1000) * 0.5 + 1.0)
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
    y = obs(torch.randn(100) * 0.5 + 1.0)
    # 4-bit asymmetric -> at most 16 distinct values
    assert torch.unique(y).numel() <= 16


def test_histogram_basic_workflow_symmetric():
    obs = HistogramObserver(bits=4, scheme=SYMMETRIC, n_bins=256)
    obs.set_mode(CALIBRATING)
    for _ in range(3):
        obs(torch.randn(1000))
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
    y = obs(torch.randn(100))
    # 4-bit symmetric -> at most 15 distinct values (zero + 7 each sign)
    assert torch.unique(y).numel() <= 15


def test_histogram_picks_clip_smaller_than_max_with_outliers():
    """KL search should clip the outlier tail when most mass is concentrated."""
    torch.manual_seed(123)
    obs_kl = HistogramObserver(bits=4, scheme=SYMMETRIC, n_bins=512)
    obs_mm = MinMaxObserver(bits=4, scheme=SYMMETRIC)

    # 99% gaussian, 1% outliers
    for _ in range(5):
        x = torch.randn(2000) * 0.3
        x[:20] = 5.0  # outliers
        obs_kl.set_mode(CALIBRATING)
        obs_kl(x)
        obs_mm.set_mode(CALIBRATING)
        obs_mm(x)
    obs_kl.freeze()
    obs_mm.freeze()

    # KL-based scale should be smaller than min/max-based (which uses the full max)
    assert obs_kl.scale.item() < obs_mm.scale.item(), (
        f"KL scale {obs_kl.scale.item()} should be < min/max scale {obs_mm.scale.item()}"
    )


def test_histogram_build_via_registry():
    obs = build_observer("histogram", bits=8, scheme=ASYMMETRIC)
    assert isinstance(obs, HistogramObserver)
    obs.set_mode(CALIBRATING)
    obs(torch.randn(500))
    obs.freeze()
    assert int(obs.mode.item()) == FROZEN
