# CLAUDE.md

Working reference for Claude Code on the **AFFNet** repo. This project's focus is **evaluation + post-training quantization** — training is out of scope.

---

## Project Overview

**Paper:** [Adaptive Frequency Filters As Efficient Global Token Mixers](https://arxiv.org/abs/2307.14008) (ICCV 2023, Microsoft Research Asia).

**Core idea:** The AFF token mixer uses the *convolution theorem* — a Hadamard product in the Fourier domain is mathematically equivalent to a convolution in the original domain. This gives **global, channel-specific, instance-adaptive token mixing in O(N log N)** with the kernel as large as the spatial resolution. The mask is *learned from the input itself*, making it semantic-adaptive.

**Variants** (all trained at 256×256):

| Variant | Params | Top-1 (ImageNet-1K) | In repo? |
|---|---|---|---|
| AFFNet-ET (`xx_small`) | 1.4M | 73.0% | ✅ Only checkpoint present |
| AFFNet-T (`x_small`) | 2.6M | 77.0% | — |
| AFFNet (`small`) | 5.5M | 79.8% | — |

Detection / segmentation are supported by the codebase but not by this workflow.

---

## Evaluation (FP32)

```bash
python main_eval.py \
  --common.config-file resource/config/imagenet_et/config.yaml \
  --common.results-loc results/ \
  --model.classification.pretrained resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt
```
Expected: **~73.02% top-1** on ImageNet-1K val (50k images).

**Assets:**
- Config: `resource/config/imagenet_et/config.yaml`
- Checkpoint: `resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt`
- Val data: `Dataset_Root/ImageNet/val/`

**Eval call chain:**
```
main_eval.py::main_worker
  → get_eval_arguments()                  [options/opts.py — reuses training parser]
  → device_setup(opts)                    [utils/common_utils.py]
  → create_eval_loader(opts)              [data/data_loaders.py]
  → get_model(opts)                       [affnet/models/__init__.py]
      → load_pretrained_model(...)        [affnet/misc/common.py]
  → Evaluator(opts, model, loader).run()  [engine/evaluation_engine.py]
```

## Best PTQ command (8/8 = 62.82% — Phase 6c reference)

```bash
python main_quant.py \
  --common.config-file resource/config/imagenet_et/config.yaml \
  --common.results-loc results/ \
  --model.classification.pretrained resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt \
  --quant.enabled \
  --quant.weight-bits 8 --quant.activation-bits 8 \
  --quant.weight-observer per_channel_min_max \
  --quant.act-observer percentile \
  --quant.quantize-linear \
  --quant.bias-correction
```

Add `--quant.fold-bn` for hardware-fidelity (Phase 7a — performance-neutral at 8/8 with per-channel weights, but matches real INT8 deployment which fuses Conv+BN). Add `--quant.insert-stubs --quant.stub-targets afno2d,affblock --quant.stub-observer min_max --quant.stub-bits 8 --quant.stub-scheme asymmetric` for the empirically "free" stub set (≈ 62.74% — the Phase 5 sweep showed `afno2d` and `affblock` are essentially free; `ln2d` and `block` cost more). Pre-wired in `.vscode/launch.json`.

---

## Architecture Reference (AFFNet-ET, 256×256)

```
(B, 3, 256, 256)
→ conv_1          ConvLayer(3→16, k3, s2)            → (B, 16, 128, 128)
→ layer_1         InvertedResidual ×1                → (B, 32, 128, 128)
→ layer_2         InvertedResidual ×3 (s2)           → (B, 48, 64, 64)
→ layer_3         AFFBlock(48→64,  2 inner Blocks)   → (B, 64, 32, 32)
→ layer_4         AFFBlock(64→104, 4 inner Blocks)   → (B, 104, 16, 16)
→ layer_5         AFFBlock(104→144, 3 inner Blocks)  → (B, 144, 8, 8)
→ conv_1x1_exp    ConvLayer(144→576, k1)             → (B, 576, 8, 8)
→ GlobalPool (mean)                                  → (B, 576)
→ LinearLayer(576→1000)                              → (B, 1000)
```

### Key modules

**InvertedResidual** (`affnet/modules/mobilenetv2.py`): MBConv = `exp_1x1 → conv_3x3 (depthwise) → red_1x1`. Residual add when `stride==1 && in==out`.

**AFFBlock** (`affnet/modules/aff_block.py:332`): stacks N inner `Block`s, then `conv_proj` (1×1) + optional `fusion` (1×1).

**Block** (`aff_block.py:272`):
```
residual = x
x = norm1(x)               # LayerNorm2D_NCHW
x = mlp(x)                 # InvertedResidual (MBConv)
if double_skip: x = x + residual; residual = x
x = norm2(x)               # LayerNorm2D_NCHW
x = filter(x)              # AFNO2D_channelfirst
x = drop_path(x)           # no-op in eval (respects .training)
x = x + residual
```

**AFNO2D_channelfirst** (`aff_block.py:62`) — the AFF token mixer:
- Input: `(B, C, H, W)` spatial features.
- Flow: `rfft2 → reshape (block-diagonal) → einsum(w1) + b1 → activation → einsum(w2) + b2 → softshrink → multiply with original spectrum → irfft2 → +bias_skip`.
- Learnable params (real-valued, leading dim 2 stacks real/imag):
  - `w1`: `(2, num_blocks, block_size, block_size * hidden_size_factor)`
  - `b1`: `(2, num_blocks, block_size * hidden_size_factor)`
  - `w2`: `(2, num_blocks, block_size * hidden_size_factor, block_size)`
  - `b2`: `(2, num_blocks, block_size)`
- ~2.3K params per Block; ~20K total across 9 Blocks (tiny vs the rest of the model).

**ConvLayer** (`affnet/layers/conv_layer.py`): `Sequential("conv", "norm", "act")` — `nn.Conv2d` always at `.block.conv`. Custom `Conv2d(nn.Conv2d)` subclass; `isinstance(m, nn.Conv2d)` matches correctly.

**Norms in this config** (`config.yaml`):
- ConvLayer norms: `sync_batch_norm`.
- Block.norm1 / Block.norm2 / AFFBlock-tail: **`LayerNorm2D_NCHW`** (`affnet/layers/normalization/layer_norm.py:80`, `nn.GroupNorm` with num_groups=1).

**Activation:** `swish` (in ConvLayer + AFNO2D internal `act`/`act2`).

---

## Config & Registry

- YAML configs flatten to dot-notation keys (`model.classification.name`) via `options/utils.py::load_config_file`.
- CLI overrides: `--key value` or `--common.override-kwargs key=value,...`.
- Eval reuses the training parser (`get_eval_arguments` → `get_training_arguments`).
- AFFNet registered via `@register_cls_models("affnet")` in `affnet/models/classification/affnet.py`.

---

## Quantization — current state (Phases 1–5 shipped)

### Approach: Q→DQ fake-quantization

Tensors stay FP32 in memory; the math simulates round-trip through a Q grid. No INT kernels required. Hardware export (TensorRT/ONNX) is out of scope.

**Math:**

*Symmetric, signed* (weights):
```
qmin = -(2^(b-1) - 1);  qmax = 2^(b-1) - 1
scale = max(|W|) / qmax        # (per-tensor)  OR  per-channel along axis 0
W_q   = clamp(round(W / scale), qmin, qmax) * scale
```

*Asymmetric, unsigned* (activations):
```
qmin = 0;  qmax = 2^b - 1
scale = (max - min) / qmax
zp    = clamp(round(qmin - min/scale), qmin, qmax)
x_q   = (clamp(round(x/scale) + zp, qmin, qmax) - zp) * scale
```

Bias stays FP32 (standard fake-quant convention).

### What is quantized today (Phase 4 architecture)

| Module | Weights | Input activations | How |
|---|---|---|---|
| Every `nn.Conv2d` (incl. `conv_1`) | ✅ | ✅ | `QuantConv2d` (per-module `weight_observer` + `act_observer`) |
| Final `nn.Linear` classifier | ✅ | ✅ | `QuantLinear` (same shape) |
| FP32 boundaries (LayerNorm2d, AFNO2D, Block, AFFBlock, ...) | — | optional | `PreStubbedModule(inner, QuantStub)` opt-in via per-type `StubConfig` |
| AFNO2D `w1`/`w2` (spectral weights) | ❌ FP32 | ❌ FP32 | not yet — Phase 6 candidate |
| Bias | ❌ FP32 | n/a | standard convention |

**On AFFNet-ET:** 56 QuantConv2d + 1 QuantLinear after conversion.

### Observer state machine

`BaseObserver` subclasses run in 3 modes: `DISABLED` (passthrough) / `CALIBRATING` (forward updates stats) / `FROZEN` (forward applies `fake_quantize` with stored scale + zero_point). Weight observers run `observe`+`freeze` once at conversion. Activation observers are calibrated on 512 shuffled ImageNet val images (16 batches × 32, seed 0), then frozen.

Registered observers (`quantization/observer.py`):
- `min_max` — per-tensor running min/max.
- `per_channel_min_max` — per-output-channel (axis 0); weight-focused, sym/asym.
- `percentile` — running mean of per-batch 0.001/0.999 quantiles; subsamples to 1M when needed.
- `per_channel_percentile` — per-channel running mean of 0.001/0.999 quantiles. Sym/asym. (Phase 7 — added, not yet swept.)
- `mse` — per-tensor; sweeps scale ∈ [0.5, 1.2] × max(|W|)/qmax, picks min-MSE. Symmetric only. (Phase 6a — does NOT help on AFFNet.)
- `per_channel_mse` — per-output-channel MSE search. Symmetric only. (Phase 6a — does NOT help on AFFNet.)
- `histogram` — streaming histogram + vectorized KL-divergence threshold search (TRT-style). 2048 bins, sym/asym. (Phase 6b — small +0.38pp lift in isolation; tied with percentile when combined with BC.)
- `per_channel_histogram` — per-channel streaming histograms; KL search vectorized across channels (loops over j only). 2048 bins, sym/asym. (Phase 7 — added, not yet swept.)

### CLI surface (`options/opts.py`, all `--quant.*`)

```
--quant.enabled              bool   (master switch; required for main_quant.py)
--quant.weight-bits          int    default 8
--quant.activation-bits      int    default 8
--quant.calib-size           int    default 512
--quant.calib-batch-size     int    default 32
--quant.calib-seed           int    default 0
--quant.skip-modules         str    default ""        (comma-sep prefixes)
--quant.weight-observer      str    default "min_max"  (or per_channel_min_max)
--quant.act-observer         str    default "min_max"  (or percentile)
--quant.weight-scheme        str    default "symmetric"
--quant.act-scheme           str    default "asymmetric"
--quant.results-csv          str    default "results/quant_sweep.csv"
--quant.quantize-linear      bool   default False
--quant.skip-linears         str    default ""
--quant.insert-stubs         bool   default False
--quant.stub-targets         str    default ""        (Phase 5: comma-sep short names)
--quant.skip-stubs           str    default ""
--quant.stub-observer        str    default "percentile"
--quant.stub-scheme          str    default "asymmetric"
--quant.stub-bits            int    default -1        (-1 → reuse activation-bits)
--quant.bias-correction      bool   default False     (Phase 6c — extra ~30s calib pass)
--quant.fold-bn              bool   default False     (Phase 7 — Conv+SyncBN/BN -> fused Conv)
--quant.stub-config          str    default ""        (Phase 7 — per-target stub override "ln2d=min_max,afno2d=percentile:8:asymmetric")
--quant.quantize-acts        bool   default False     (Phase 8b — replace Swish/HardSwish with QuantHardswish; NEGATIVE result, see Phase 8b)
--quant.skip-acts            str    default ""
```

**Phase 5 stub target registry** (in `main_quant.py::_STUB_TARGET_REGISTRY`, keyed by short name):
- `ln2d` → `LayerNorm2D_NCHW` (21 sites: Block.norm1, Block.norm2, AFFBlock tail-LN)
- `afno2d` → `AFNO2D_channelfirst` (9 sites)
- `block` → `Block` (9 sites)
- `affblock` → `AFFBlock` (3 sites)

### CSV layout

- `results/quant_sweep.csv` — Phase 1/2 history (no QuantLinear, no stubs).
- `results/quant_sweep_v2.csv` — Phase 3+ history (auto-routed when `--quant.quantize-linear`, `--quant.insert-stubs`, or `--quant.quantize-acts` is on).

### Files

```
quantization/
├── fake_quant.py        — fake_quantize(x, scale, zp, qmin, qmax)
├── observer.py          — BaseObserver + 8 registered observers (per-tensor & per-channel
│                          variants of {min_max, percentile, mse, histogram})
├── quant_conv.py        — QuantConv2d
├── quant_linear.py      — QuantLinear (refuses GroupLinear and channel_first=True)
├── quant_stub.py        — QuantStub, PreStubbedModule, StubConfig
├── quant_hardswish.py   — QuantHardswish (Phase 8b — function swap + input quant in one)
├── convert.py           — convert_model(...), _swap_conv2d, _swap_linears, _swap_acts, _insert_stubs
├── calibrate.py         — calibrate(...), collect_quant_modules (duck-typed on .act_observer)
├── bias_correction.py   — apply_bias_correction (Phase 6c)
├── fuse.py              — fold_conv_bn_(model) (Phase 7a; deletes BN child after Phase 8a)
└── tests/               — pytest
main_quant.py            — entry point; _STUB_TARGET_REGISTRY + _build_stub_targets
                          + _build_stub_targets_from_config (Phase 7c)
```

---

## Observed PTQ results (AFFNet-ET, ImageNet-1K val 50k, FP32 baseline 73.02%)

### Conv2d-only (Phase 1/2, `quant_sweep.csv`)

| w/a | w_observer | a_observer | top-1 | note |
|---|---|---|---|---|
| 16/16 | min_max | min_max | 72.95 | math sanity |
| 8/8 | min_max | min_max | 0.29 | per-tensor min/max collapses (Swish outliers) |
| 8/8 | min_max | percentile | 29.79 | percentile recovers +30pt |
| 8/8 | per_channel_min_max | percentile | **59.50** | Phase 2 best |
| 4/8, 8/4, 4/4 | per_channel_min_max | percentile | ~0.1 | 4-bit collapse |

### Conv2d + Linear (Phase 4, `quant_sweep_v2.csv`)

Phase 4 added per-Conv `act_observer` (replacing Phase 3's coarse block-boundary stubs), restored `conv_1` to the quantized set, and wrapped the final Linear classifier.

| w/a | w_observer | a_observer | top-1 | note |
|---|---|---|---|---|
| 16/16 | per_channel_min_max | min_max | 72.97 | math sanity ✓ |
| 16/16 | per_channel_min_max | percentile | 69.95 | percentile clips real data at 16-bit |
| 8/8 | per_channel_min_max | **percentile** | **59.38** | Phase 4 best 8/8 |
| 8/8 | min_max | min_max | 0.32 | per-tensor min/max still collapses |

Phase 4 best 8/8 (59.38) ≈ Phase 2 best (59.50) despite extending coverage to `conv_1` + classifier. Per-Conv act observers were +1.77pp better than Phase 3's coarser block-boundary stubs at 8/8 (57.61 → 59.38).

### Phase 6 results (2026-04-27)

**Phase 6a — MSE-based weight observers (`mse`, `per_channel_mse`):**

| w/a | w_observer | a_observer | top-1 | vs baseline |
|---|---|---|---|---|
| 8/8 | per_channel_mse | percentile | 58.86 | -0.52 (slightly worse than per_channel_min_max) |
| 8/8 | mse | percentile | 25.00 | per-tensor MSE collapses similarly to min_max |
| 4/8 | per_channel_mse | percentile | 0.18 | 4-bit cliff unchanged |
| 4/4 | per_channel_mse | percentile | 0.10 | 4/4 cliff unchanged |

**Conclusion:** MSE-optimal scale doesn't help on AFFNet weights. Per-channel min/max is essentially optimal at 8-bit, and 4-bit cliff is fundamental, not a scale-selection issue. **Negative result documented.**

**Phase 6b — Histogram + KL-divergence observer (`histogram`):**

TensorRT-style: streaming histogram, vectorized KL search at freeze (n_bins=2048 default, ~0.8s per layer). Activation-side; works with both symmetric and asymmetric.

| w/a | a_observer | bias_corr | top-1 | vs comparable percentile |
|---|---|---|---|---|
| 8/8 | histogram | off | 59.76 | percentile (off): 59.38 → **+0.38** |
| 8/8 | histogram | on  | 62.60 | percentile (on): 62.82 → -0.22 (tie) |
| 8/4 | histogram | on  | 0.13  | percentile (on): 0.12 — cliff unchanged |

**Conclusion:** Histogram is marginally better than percentile in isolation (+0.38pp at 8/8) but doesn't compose with bias correction — BC absorbs most of the same gain. Doesn't break the 4-bit activation cliff. Useful as a drop-in option for when BC isn't available.

**Phase 6c — bias correction (`--quant.bias-correction`):**

For each Conv/Linear with FP32 weights `W`, quantized weights `W_q`, bias `b`:
```
delta_b = layer(E[x], W - W_q)        # E[x] measured post-quant on calib set
b' = b + delta_b
```

| w/a | bias_corr | top-1 | Δ |
|---|---|---|---|
| 8/8 | off | 59.38 | (Phase 4 best) |
| 8/8 | **on**  | **62.82** | **+3.44** |
| 4/8 | on  | 0.11 | cliff unchanged |
| 8/4 | on  | 0.12 | cliff unchanged |

**Conclusion:** Bias correction is the biggest single 8/8 lift in the project so far. Compensates the systematic per-channel mean shift introduced by weight quantization. Cheap (~30s extra calibration pass over 16 batches). **Best 8/8 = 62.82%, gap to FP32 narrowed from 13.6pp → 10.2pp.** Does not help at 4-bit — the cliff is rounding/representation, not a mean shift.

**Combined Phase 5 + 6c interactions at 8/8:**

| Config | top-1 | vs BC alone (62.82) |
|---|---|---|
| BC + no stubs | 62.82 | (reference) |
| BC + `afno2d,affblock` stubs (Phase 5 "free" set) | 62.74 | -0.08 |
| BC + all 4 stub targets (`ln2d,afno2d,block,affblock`) | 60.56 | -2.26 |

Stubs and BC stack additively to first order; the marginal stub cost matches the no-BC Phase 5 numbers. Phase 5's "free" stub set (`afno2d,affblock`) is the right tradeoff if hardware fidelity matters; the full 4-target set is ~2.3 pp expensive for the extra coverage.

**Phase 6 takeaway:** Better observers don't help at low bit-widths on AFFNet, but **mathematical bias compensation** does. The 4-bit cliff requires AdaRound / QAT.

### Phase 7 results (2026-04-30)

**Phase 7a — Conv+BN folding (`--quant.fold-bn`):**

Standard fusion: `α = γ/√(σ²+ε); W' = α·W; b' = α·b + β − μ·α`. BN child is deleted from its parent `Sequential` (Phase 8a refinement; was `nn.Identity()` placeholder originally). AFFNet-ET has 56 Conv→SyncBN pairs (every Conv layer has one). Folding happens BEFORE `convert_model` so the QuantConv2d quantizes the FOLDED weights as a single fused op.

| Config | top-1 | Δ |
|---|---|---|
| FP32 no fold (T1) | 72.958 | (reference) |
| FP32 + fold (T2) | 72.928 | -0.030 (= float-32 conv kernel noise — fold is mathematically lossless) |
| 8/8 + BC, no fold (T3) | 62.816 | (matches Phase 6c reference 62.82) |
| 8/8 + BC + fold (T4) | 62.766 | -0.050 vs T3 (= run-to-run noise) |

**Conclusion:** **BN folding is performance-neutral at 8/8 with `per_channel_min_max` weights.** Reason: per-channel weight quant already gives each output channel its own scale, so multiplying weights by α before quantization just rescales each channel uniformly — the per-channel scale absorbs α perfectly. With per-tensor weight quant (`min_max`), folding *would* matter (untested but predicted to help significantly).

**Why ship it anyway:** hardware fidelity. Real INT8 deployment (TensorRT/ONNX Runtime/TFLite) fuses Conv+BN automatically. Default off (preserves Phase 1–6 behavior).

**Phase 7b — per-channel observers (`per_channel_percentile`, `per_channel_histogram`):**

Registered, not yet swept. Available as `--quant.weight-observer per_channel_percentile` (sym/asym) and `--quant.weight-observer per_channel_histogram` (sym/asym, KL search vectorized across channels). Likely interesting at 4-bit weights where per-channel min/max wastes resolution on outliers.

**Phase 7c — per-target stub config (`--quant.stub-config`):**

Format: `name=observer[:bits[:scheme]],...` e.g. `ln2d=min_max,afno2d=histogram:8:asymmetric`. Lets you mix observers across stub targets instead of one global choice. Implemented; no sweep yet.

### Phase 8 results (2026-05-01)

**Phase 8a — drop the `nn.Identity` placeholder after BN folding:**

`fold_conv_bn_` previously installed `nn.Identity()` in place of the folded BN to keep `Sequential` indices intact. With no caller depending on the placeholder (verified by grep across the repo), the BN child is now `del`'d outright. `Sequential.forward` iterates `_modules.values()` so removed entries are skipped cleanly; insertion order of the remaining children is preserved by the underlying `OrderedDict`. **Math-preserving cleanup; no eval impact** (synthetic test confirms output matches within float32 noise).

**Phase 8b — `QuantHardswish` (function swap + activation-input quant fused) — NEGATIVE result:**

New module `quantization/quant_hardswish.py` mirrors `QuantConv2d` / `QuantLinear`: own `act_observer` on its input, fake-quants the input, then runs `F.hardswish`. The convert pass (`_swap_acts` in `convert.py`, behind `--quant.quantize-acts`) replaces every `nn.SiLU` / `Swish` / `nn.Hardswish` / `Hardswish` site with `QuantHardswish` — bundling the function swap and the activation-input quantization into one transformation, matching real INT8 deployment where the Conv output is requantized before the nonlinearity reads it.

**Site coverage on AFFNet-ET:** 55 sites (37 ConvLayer.act + 18 AFNO2D internal `act`/`act2`). Note: only 37 of 56 ConvLayers have an `act` submodule — the rest are `use_act=False`, so the earlier "56 ConvLayer.act" estimate was wrong.

| # | Config | top-1 | Δ |
|---|---|---|---|
| R1 | FP32 Swish (reference) | 73.02 | — |
| R2 | FP32 HardSwish (`--common.override-kwargs model.classification.activation.name=hard_swish`) | **35.17** | **−37.85 vs R1** (function swap alone in FP32) |
| R3 | 8/8 + BC, Swish (Phase 6c reference) | 62.82 | — |
| R4 | 8/8 + BC + `--quant.quantize-acts` | **0.42** | **−62.40 vs R3** |

**Decomposition:** function swap costs 37.85pp in FP32 (R1 → R2); adding activation-input quant on top of the swapped HardSwish costs another ~34.75pp (R2 → R4). Both contributions are independently fatal.

**Conclusion:** Replacing Swish with HardSwish on a Swish-trained model is not viable without retraining — the function-shape mismatch alone halves accuracy in FP32. The integrated swap is therefore not the right hardware-fidelity path to ship.

**What this rules in/out for next steps:**
- Activation-input quantization (alone) is *not* the killer here, but we never tested it on Swish (the integrated `QuantHardswish` bundles the swap). A separate `QuantSwish` (input-quant only, function unchanged) would isolate the input-quant cost on the right baseline. Open question — not yet wired.
- HardSwish substitution is dead unless we retrain (out of scope per CLAUDE.md). Don't burn more compute on this branch.
- The FP32-HardSwish loss is much larger than typical post-hoc activation swaps (literature suggests <5pp). Worth understanding *why* AFFNet is so sensitive — likely the AFNO2D spectral path's `act`/`act2` are the contributors (Swish smoothness around 0 may matter for the Hadamard-multiplied spectrum), but un-investigated.

**Code shipped:** `quantization/quant_hardswish.py` + `_swap_acts` + `--quant.quantize-acts` / `--quant.skip-acts` + 8 unit tests + diagnostic in `main_quant.py`. Default OFF, so existing pipelines (Phases 1–7) are untouched.

**Phase 8c — leaf-stub sweep (8/8 + BC + stubs on every leaf except Conv/BN/Linear) — NEGATIVE result:**

Two new short names added to `_STUB_TARGET_REGISTRY` (`main_quant.py`):
- `swish` → `Swish` (covers ConvLayer.act sites + AFNO2D internal `act`/`act2`)
- `globalpool` → `GlobalPool` (the pre-classifier head)

Run config: `--quant.insert-stubs --quant.stub-targets swish,ln2d,afno2d,globalpool --quant.stub-observer min_max --quant.stub-scheme asymmetric`. **86 stubs installed: 55 swish + 21 ln2d + 9 afno2d + 1 globalpool.**

Result: **R5 = 3.63% top-1** (vs R3 = 62.82). −59pp from BC@8/8.

**Why min_max ≠ right answer for Swish stubs:** Phase 5's "min_max dominates percentile for stubs" finding holds only for *post-norm/residual* tensors (tightly bounded). Swish-input stubs see *raw Conv output* (= post-BN-fold), which has wide range with outliers — the diagnostic showed `min_val` running averages spanning [-45, -5]. `min_max` lets outliers blow up the scale, coarsening the bulk. The right observer for Swish stubs is `percentile`. Phase 5's stub sweep included `block` (which sees post-residual, tightly bounded) but not `swish` directly, so this distinction wasn't visible until now.

**Untested follow-up (~10 min compute):** Use Phase 7c per-target stub config:
```
--quant.stub-config swish=percentile,ln2d=min_max,afno2d=min_max,globalpool=min_max
```
Hypothesis: most of the −59pp lives in the 55 swish stubs; the other 31 (ln2d+afno2d+globalpool) cost roughly Phase 5 levels (≤2pp).

### Phase 5 — FP32-boundary stub sweep (2026-04-27)

Each row = Phase 4 8/8 baseline + `--quant.insert-stubs` with one (or more) target classes. `Δ` is vs Phase 4 baseline 59.38%.

| Stub target | #stubs | min_max obs | Δ | percentile obs | Δ |
|---|---|---|---|---|---|
| `ln2d` (LayerNorm2D inputs)        | 21 | 58.64 | -0.74 | 57.93 | -1.45 |
| `afno2d` (AFNO2D inputs)           |  9 | **59.34** | **-0.04** | 59.17 | -0.21 |
| `block` (Block inputs)             |  9 | 59.04 | -0.34 | 57.96 | -1.42 |
| `affblock` (AFFBlock inputs)       |  3 | **59.40** | **+0.02** | 58.85 | -0.53 |
| **all 4 combined**                 | 42 | 57.72 | -1.66 | — | — |

**Findings:**

1. **min_max dominates percentile for stubs** — opposite of the per-Conv path. Stubs see post-norm/residual tensors (tightly bounded); percentile clipping discards real data while min_max captures it cleanly. **Do not blindly reuse percentile for stubs.**
2. **The architecture is robust to FP32-boundary stubs at 8/8.** Worst single target (`ln2d`) is -0.74pp; all four together are -1.66pp; `afno2d` and `affblock` stubs are essentially free.
3. **The 13.6pp gap to FP32 is NOT coming from FP32 boundaries.** Even forcing 42 stubs onto the Q grid only costs 1.66pp. The gap lives in the per-Conv weight/activation observers — fixing it needs better observers, not more boundary coverage.
4. **`PreStubbedModule` quantizes the INPUT of the inner module**, modeling a fused-kernel int8 memory read. So `ln2d` target = quantize the residual flowing INTO every LayerNorm2d, etc.

### Where FP32 still lives (Phase 5 still hasn't covered)

- AFNO2D internals: `rfft2 → einsum(w1) → activation → einsum(w2) → softshrink → multiply with original spectrum → irfft2`. `w1`, `w2`, `b1`, `b2` are FP32. Spectral activations are complex-valued (real/imag stacked).
- Residual `+` adds — covered indirectly because the next QuantConv2d's `act_observer` captures the post-add distribution. The only `+` not so covered is `Block.x + residual` at `aff_block.py:309`, which feeds the *next* Block's `norm1` (LayerNorm2d, FP32) — caught by `--quant.stub-targets ln2d` if desired.
- Bias tensors and the final softmax (eval doesn't apply softmax explicitly).

### Dropout in eval — not a bug

`DropPath` (`aff_block.py:46-59`) checks `self.training` and is a no-op in eval. Standard `nn.Dropout` similarly. The visible `x = self.drop_path(x)` line in `Block.forward` is dead code in eval mode — investigated and confirmed during Phase 5 exploration.

---

## Phase 8 candidates (where to attack next)

Best 8/8 to date = **62.82%** (Phase 6c, bias correction; fold-bn neutral at 8/8). Gap to FP32 = 10.2pp. 4-bit configs still collapse.

Remaining levers, ordered by expected payoff vs effort:

1. **AdaRound / BRECQ** — the standard answer to the 4-bit weight cliff. Learns rounding direction per weight via per-layer reconstruction loss. Requires a per-layer optimizer loop (~500 lines). Expected lift at 4/8: from ~0% to 50–60%.
2. **Sweep the new per-channel observers** (`per_channel_percentile`, `per_channel_histogram`) at 8/8 and 4/8. Cheap (~30 min compute, no new code). Phase 7b registered them but didn't run them.
3. **SmoothQuant** — per-channel activation scaling absorbed into adjacent weights. Useful if 8/4 or 4/4 are interesting; ~100 lines.
4. **Heterogeneous stub sweep via `--quant.stub-config`** — verify the global `min_max` stub-observer choice from Phase 5 is optimal across targets, or find tiny extra lift. Cheap.
5. **Quantize AFNO2D `w1`/`w2`** — adds coverage to currently-FP32 spectral path. Tiny param count (20K), unclear marginal accuracy. Needs complex-aware observer + new `QuantEinsum` module.
6. **Per-layer mixed precision** — sensitivity analysis to assign more bits to critical layers. Cheap diagnostic, then surgical bit-width assignment.
7. **Per-tensor weight observer + fold-bn** — Phase 7a predicts folding would actually help here (per-tensor scale wastes resolution without folding because BN's α inflates some channels). Untested. Cheap A/B.

---

## Training (not the workflow focus)

```bash
python main_train.py --log-wandb --common.config-file <config> --common.results-loc <save>
```
Engine: `engine/training_engine.py` (~1600-line Trainer class). Defer to this section only if asked.

---

## Environment

- Python 3.8 in conda env `AFFNet`.
- **Active Python:** `C:\Users\alona\miniconda3\envs\AFFNet\python.exe` (base env is missing key deps like `pypdf`, `complexPyTorch`).
- Key deps: `torch==1.13.1`, `torchvision==0.15.2`, `complexPyTorch==0.4`, `torch-dct==0.1.6`.
- No `setup.py` — run in-place from the project root.
- Paper PDF: `C:\Users\alona\Projects\TokenMixers\AFFNet\docs\adaptive freq filters.pdf` (outside this repo).
