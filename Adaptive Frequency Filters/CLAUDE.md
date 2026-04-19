# CLAUDE.md

Working reference for Claude Code on the **AFFNet** repo. This project's focus is **evaluation + post-training quantization** — training is out of scope for this workflow.

---

## Project Overview

**Paper:** [Adaptive Frequency Filters As Efficient Global Token Mixers](https://arxiv.org/abs/2307.14008) (ICCV 2023, Microsoft Research Asia).

**Core idea:** The AFF token mixer uses the *convolution theorem* — a Hadamard product in the Fourier domain is mathematically equivalent to a convolution in the original domain. This realizes **global, channel-specific, instance-adaptive token mixing in O(N log N)** (vs O(N²) for self-attention), with the kernel effectively as large as the spatial resolution. The mask is *learned from the input itself*, making it semantic-adaptive.

**Ships three variants** (all trained at 256×256):
| Variant | Params | Top-1 (ImageNet-1K) | In repo? |
|---|---|---|---|
| AFFNet-ET (`xx_small`) | 1.4M | 73.0% | ✅ Only checkpoint present |
| AFFNet-T (`x_small`) | 2.6M | 77.0% | — |
| AFFNet (`small`) | 5.5M | 79.8% | — |

Also supports detection (SSD on COCO) and segmentation (DeepLabv3 on ADE20K/VOC), but this workflow is classification-only.

---

## Evaluation (primary workflow)

**Command:**
```bash
python main_eval.py \
  --common.config-file resource/config/imagenet_et/config.yaml \
  --common.results-loc results/ \
  --model.classification.pretrained resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt
```
Expected: **~73.02% top-1** on ImageNet-1K val.

**Assets in the repo:**
- Config: `resource/config/imagenet_et/config.yaml` (only one)
- Checkpoint: `resource/model/imagenet_et/checkpoint_ema_score_73.0238.pt` (only one)
- Val data: `Dataset_Root/ImageNet/val/` (standard ImageNet folder layout)

**Eval call chain:**
```
main_eval.py::main_worker
  → get_eval_arguments()                  [options/opts.py — reuses training parser]
  → device_setup(opts)                    [utils/common_utils.py]
  → create_eval_loader(opts)              [data/data_loaders.py]
      → evaluation_datasets(opts)         [data/datasets/__init__.py, DATASET_REGISTRY]
      → build_sampler(is_training=False)
  → get_model(opts)                       [affnet/models/__init__.py]
      → build_classification_model(opts)  [affnet/models/classification/__init__.py]
      → load_pretrained_model(...)        [affnet/misc/common.py]
  → Evaluator(opts, model, loader)        [engine/evaluation_engine.py]
  → Evaluator.run() → eval_fn_image()     [no_grad loop, metric aggregation]
```

**Key eval files:**
| File | Role |
|---|---|
| `main_eval.py` | Entry point |
| `engine/evaluation_engine.py` | `Evaluator` class, eval loop |
| `affnet/misc/common.py` | `load_pretrained_model()` — handles rename/exclude scopes |
| `data/data_loaders.py` | `create_eval_loader()` |
| `metrics/metric_monitor.py` | top1/top5 aggregation (DDP-aware) |
| `options/opts.py` | All CLI/YAML arg definitions |

---

## Architecture Reference

**Full forward shape** (`small` / AFFNet, 256×256 input):
```
(B, 3, 256, 256)
→ conv_1          ConvLayer(3→16, k3, s2)          → (B, 16, 128, 128)
→ layer_1         InvertedResidual ×1 (s1, exp=4)   → (B, 32, 128, 128)
→ layer_2         InvertedResidual ×3 (s2, exp=4)   → (B, 64, 64, 64)
→ layer_3         AFFBlock(64→128, 2 inner blocks)  → (B, 128, 32, 32)
→ layer_4         AFFBlock(128→256, 4 inner blocks) → (B, 256, 16, 16)
→ layer_5         AFFBlock(256→320, 3 inner blocks) → (B, 320, 8, 8)
→ conv_1x1_exp    ConvLayer(320→1280, k1)           → (B, 1280, 8, 8)
→ GlobalPool (mean)                                 → (B, 1280)
→ LinearLayer(1280→num_classes)                     → (B, 1000)
```

**Variant channel configs (L3/L4/L5):**
| Scale | L3 | L4 | L5 | Exp |
|---|---|---|---|---|
| xx_small (AFFNet-ET) | 48→64 | 64→104 | 104→144 | 576 |
| x_small (AFFNet-T) | 48→96 | 96→160 | 160→192 | 768 |
| small (AFFNet) | 64→128 | 128→256 | 256→320 | 1280 |
| base / large | larger — see `affnet/models/classification/config/affnet.py` | | | |

### Key modules

**InvertedResidual** (MobileNetV2, `affnet/modules/mobilenetv2.py`):
```
exp_1x1 [Conv+BN+Act] → conv_3x3 depthwise [Conv+BN+Act] → red_1x1 [Conv+BN]
+ residual when stride=1 and in_channels==out_channels
```
Used in layers 1–2 and as the channel mixer (MBConv) inside AFFBlock.

**AFFBlock** (`affnet/modules/aff_block.py`): stacks N inner `Block`s:
```
Block = LayerNorm2d → MBConv (channel mixer) +skip → LayerNorm2d → AFNO2D_channelfirst (token mixer) +skip
```
Then `conv_proj (k1)` + optional `fusion (k1)` to combine with the pre-block skip feature.

**AFNO2D_channelfirst** — the *AFF token mixer* (core of the paper):
- Input: `(B, C, H, W)` spatial features
- Flow: `rfft2 → group einsum (w1) + ReLU → group einsum (w2) → softshrink → elementwise multiply with original spectrum → irfft2`
- The first einsum+ReLU+second einsum is the **mask subnetwork** M(·); `M ⊙ F(X)` is the convolution-theorem trick
- Learnable params: `w1, w2` shape `(2, num_blocks, block_size, block_size)`, `b1, b2` shape `(2, num_blocks, block_size)` — the leading `2` is real/imag components, `num_blocks=8` by default (block-diagonal for efficiency)
- Soft-shrink provides sparsity regularization

**ConvLayer** (`affnet/layers/conv_layer.py`): thin wrapper over `Conv2d + [Norm] + [Act]`. Will be the main target for BN-folding during quantization.

**Primitives:** `GlobalPool` (mean/rms/abs), `LinearLayer`, `LayerNorm2d`, activation variants (swish, gelu, hard_swish, …) all under `affnet/layers/`.

---

## Config & Registry System

- YAML configs are flattened to dot-notation keys (`model.classification.name`) via `options/utils.py::load_config_file`
- CLI overrides: `--key value` or `--common.override-kwargs key=value,...`
- All args defined in `options/opts.py`; eval uses the same parser as training (`get_eval_arguments` → `get_training_arguments`)
- **Registry pattern** — models/datasets/metrics registered by name at import time:
  - `TASK_REGISTRY[dataset.category](opts)` → `CLS_MODEL_REGISTRY[model.classification.name](opts)`
  - AFFNet registered via `@register_cls_models("affnet")` in `affnet/models/classification/affnet.py`

---

## Quantization (in progress — Phase 1)

**Goal:** Post-Training Quantization via **fake quantization (Q→DQ simulation)** — arbitrary bit-widths (16/8/6/4) for research without needing hardware INT kernels. Tensors stay FP32 in memory; we simulate the quantization error and measure the accuracy hit using the existing `Evaluator`. Hardware export (TensorRT/CoreML/ONNX) is out of scope for this workflow.

### Phase 1 scope (current)

`nn.Conv2d` layers only, both **weights** (static) and **input activations** (calibrated). Non-Conv2d tensors — LayerNorm2d outputs, residual adds, the AFNO2D spectral path (`rfft2`/einsum/`irfft2`), GlobalPool, final Linear — stay FP32. Phase 1 is *not* a full whole-chain hardware simulation.

Decisions locked in:
- Split bits: `--quant.weight-bits` and `--quant.activation-bits` independently
- Weights: **symmetric**, per-tensor, derived statically from the weight tensor itself
- Activations: **asymmetric**, per-tensor, calibrated on 512 shuffled ImageNet val images
- Skip-list: `conv_1` (first Conv, sees raw RGB) stays FP32 — module-path prefix via `--quant.skip-modules`
- Bias stays FP32 (standard fake-quant convention)

### Fake-quant math

**Symmetric, signed** (weights, `b` bits) — zero_point ≡ 0:
```
qmin = -(2^(b-1) - 1);  qmax = 2^(b-1) - 1
scale = max(|W|) / qmax
W_q   = clamp(round(W / scale), qmin, qmax) * scale
```

**Asymmetric, unsigned** (activations, `b` bits):
```
qmin = 0;  qmax = 2^b - 1
scale = (max - min) / qmax
zp    = clamp(round(qmin - min/scale), qmin, qmax)
x_q   = (clamp(round(x/scale) + zp, qmin, qmax) - zp) * scale
```

### Planned file layout

```
quantization/
├── __init__.py
├── fake_quant.py         # fake_quantize(x, scale, zp, qmin, qmax)
├── observer.py           # BaseObserver + MinMaxObserver + OBSERVER_REGISTRY
├── quant_conv.py         # QuantConv2d(nn.Module) wrapping nn.Conv2d
├── convert.py            # convert_model(model, cfg): walk tree, swap, apply skip-list
├── calibrate.py          # calibrate(model, loader, n_batches): forward + freeze observers
└── tests/                # pytest — unit per module + integration
main_quant.py             # Entry point (mirrors main_eval.py)
results/quant_sweep.csv   # Appended per run: w_bits,a_bits,calib_size,w_obs,a_obs,top1,top5,n_convs
```

### Observer state machine

Every `BaseObserver` subclass has three modes: `DISABLED` (passthrough), `CALIBRATING` (forward updates stats), `FROZEN` (forward applies `fake_quantize` with stored `scale`/`zero_point`). Weight observers run `observe` + `freeze` once in `QuantConv2d.__init__`. Activation observers are set to `CALIBRATING` for the calibration pass, `FROZEN` for eval.

This is the single extensibility seam — Phase 2 observers (`PerChannelMinMaxObserver`, `PercentileObserver`, `MSEObserver`, `HistogramObserver`) are new subclasses selected via `--quant.*-observer`; no pipeline changes.

### CLI surface (register in `options/opts.py`)

```
--quant.enabled              bool,  default False   (keeps main_eval untouched)
--quant.weight-bits          int,   default 8
--quant.activation-bits      int,   default 8
--quant.calib-size           int,   default 512
--quant.calib-batch-size     int,   default 32
--quant.calib-seed           int,   default 0
--quant.skip-modules         str,   default "conv_1"  (comma-sep prefixes)
--quant.weight-observer      str,   default "min_max"
--quant.act-observer         str,   default "min_max"
--quant.weight-scheme        str,   default "symmetric"
--quant.act-scheme           str,   default "asymmetric"
```

### Success gates (sweep on AFFNet-ET vs 73.02% FP32 baseline)

| Config | Expected top-1 | Meaning if it misses |
|---|---|---|
| `w=16, a=16` | 73.02% ± 0.1% | **Hard gate** — Q→DQ math is buggy, stop and debug |
| `w=8, a=8` | ~72.5–73.0% | Standard 8-bit PTQ territory |
| `w=4, a=8` | ~70–72% | Weight sensitivity — depthwise convs likely suffer |
| `w=8, a=4` | ~65–72% | Activation sensitivity — late-layer outliers likely suffer |
| `w=4, a=4` | <70% expected | Baseline for Phase 2 improvements |

### Phase 1 + Phase 2 observed results (2026-04-18/19)

Full sweep on AFFNet-ET (ImageNet val, 50k images, FP32 baseline 72.95%):

| w/a | w_observer | a_observer | top-1 | note |
|---|---|---|---|---|
| 16/16 | min_max | min_max | **72.95** | Phase 1 hard gate ✓ |
| 16/16 | per_channel_min_max | percentile | 69.98 | percentile clips real data at 16b |
| 8/8 | min_max | min_max | 0.29 | Phase 1 cliff (outlier sensitivity) |
| 8/8 | min_max | percentile | 29.79 | percentile alone: +30pt lift |
| 8/8 | **per_channel_min_max** | **percentile** | **59.50** | Phase 2 best 8/8 |
| 4/8 | per_channel_min_max | percentile | 0.15 | 4-bit weights collapse |
| 8/4 | per_channel_min_max | percentile | 0.11 | 4-bit activations collapse |
| 4/4 | per_channel_min_max | percentile | 0.10 | 4/4 collapse |

**Takeaways:**
- Phase 1 Q→DQ math is correct (16/16 gate passes).
- The original "8/8 → ~72.5%" expectation was optimistic for per-tensor min/max on this architecture's swish+MBConv path. Real Phase 1 baseline is **0.29%** at 8/8.
- **Percentile** activation observer contributes **+30pt** at 8/8; **per-channel** weight observer contributes another **+30pt** on top. Combined: **59.5%** at 8/8 (13.5pt gap vs FP32).
- **Do NOT combine percentile with 16/16** — it drops 3pt vs min_max because 16-bit can represent real outliers losslessly.
- **Any 4-bit tensor (w or a) still collapses** with current observers. Phase 3 candidates: MSEObserver (weights), HistogramObserver/KL (activations), bias correction, AdaRound.

### Build order

1. `fake_quant.py` + pytest (round-trip math)
2. `observer.py` + pytest (MinMax stats + freeze)
3. `quant_conv.py` + pytest (matches `nn.Conv2d` in `DISABLED` mode)
4. `convert.py` + `calibrate.py`
5. `main_quant.py` + register `--quant.*` flags
6. Run 16/16 gate → 8/8 → 4-bit sweep → record CSV

### Known Phase 1 weaknesses (anticipate)

- **Depthwise convs + per-tensor weights** → likely 4-bit accuracy cliff. `InvertedResidual.conv_3x3` uses `groups=in_channels`; each output channel has its own filter with wildly different magnitude, and per-tensor `max(|W|)` crushes small channels. Phase 2 priority #1.
- **Swish/GELU outlier activations** in layers 3–5 → min/max calibration captures rare spikes, inflates scale, loses resolution. Phase 2 priority #2.
- **Residual adds and AFNO2D spectral path stay FP32** — not a whole-chain hardware simulation. Phase 2+ item.

### Phase 2 roadmap (each a new `BaseObserver` subclass)

1. ✅ `PerChannelMinMaxObserver` (weights, per output channel) — shipped in `observer.py`, name `per_channel_min_max`. Reshape-on-broadcast fake_quantize. Wired via `--quant.weight-observer per_channel_min_max`.
2. ✅ `PercentileObserver` (activations, 99.9%) — shipped in `observer.py`, name `percentile`. Running-mean-of-per-batch-quantiles; subsamples to 1M when tensor is larger.
3. `MSEObserver` — scale minimizing MSE between FP and quantized weight
4. `HistogramObserver` + KL-divergence — TensorRT-style calibration
5. ✅ Extend coverage: wrap `LinearLayer` (`QuantLinear`), add quant stubs at block boundaries (`QuantStub` / `StubbedModule`). **AFNO2D spectral weights still deferred.**
6. Per-layer bit-width mixing (sensitivity analysis → assign more bits to critical layers)
7. AdaRound / BRECQ — learn rounding direction, not just nearest
8. Bias correction — per-layer bias shift from compounding activation-quant error

---

## Phase 3 — Hardware-faithful PTQ simulation (in progress)

**Why:** Phase 1/2 only fake-quantizes Conv2d weights and inputs; between blocks, values return to FP32 (BN, activations, residual adds, AFNO2D output, GlobalPool, the final `LinearLayer`). Deployed INT hardware does *not* dequantize between layers — the quantization grid carries across. Phase 3 inserts explicit re-quantization points (`QuantStub`) at block boundaries and wraps the classifier head with `QuantLinear`, so the simulation models the actual data flow.

### What shipped

- `quantization/quant_stub.py` — `QuantStub(nn.Module)` wraps a `build_observer(...)` and is driven by the usual DISABLED / CALIBRATING / FROZEN state machine; `StubbedModule(inner, stub)` wraps a target module so `inner(x) → stub(...)` happens transparently.
- `quantization/quant_linear.py` — `QuantLinear` mirrors `QuantConv2d` for `nn.Linear` and AFFNet's `LinearLayer(channel_first=False)`. Refuses `GroupLinear` and `channel_first=True` (the latter routes through `F.conv2d` and is Conv-territory).
- `quantization/convert.py` — `convert_model` gains opt-in `quantize_linear`, `insert_stubs`, `skip_linears`, `skip_stubs`, `stub_bits`, `stub_observer`, `stub_scheme` kwargs. Default `stub_targets` are `InvertedResidual`, `InvertedResidualSE`, `Block`, `AFFBlock`, `GlobalPool` (captures every FP32 boundary on AFFNet-ET).
- `quantization/calibrate.py` — generalized to `collect_quant_modules`: anything with a `BaseObserver`-typed `.act_observer` buffer (QuantConv2d + QuantLinear + QuantStub + future).
- `main_quant.py` — threads the new flags, extends the diagnostic dump, and routes Phase 3 runs to `results/quant_sweep_v2.csv` so the Phase 1/2 history stays intact.

### CLI surface (additions, all default OFF)

```
--quant.quantize-linear      bool,  default False   # swap LinearLayer → QuantLinear
--quant.skip-linears         str,   default ""      # comma-sep prefixes
--quant.insert-stubs         bool,  default False   # wrap block-boundary modules
--quant.skip-stubs           str,   default ""      # comma-sep prefixes
--quant.stub-observer        str,   default "percentile"
--quant.stub-scheme          str,   default "asymmetric"
--quant.stub-bits            int,   default -1      # -1 → reuse --quant.activation-bits
```

### Block-boundary stub policy

Default `stub_targets` wrap every instance of:
- `InvertedResidual` — catches the `x + self.block(x)` residual and the bare-sequential path (`use_res_connect=False`). Also catches the adds inside every `Block.mlp`.
- `Block` (`affnet/modules/aff_block.py:272`) — catches the `x + residual` after `filter(AFNO2D)` (9 sites total: 2+4+3 across layer_3/4/5).
- `AFFBlock` — stage output before it feeds the next layer.
- `GlobalPool` — pooled vector before the classifier head.

### Verification gates

1. **Unit tests:** `pytest quantization/tests -q` — 100+ tests passing.
2. **Regression:** defaults (Phase 3 flags all False) rerun of `w=8 a=8 per_ch + percentile` → expect within 0.1pp of 59.5%.
3. **3a (Linear) 16/16 math-sanity:** `--quant.quantize-linear` + w=16 a=16 per_ch + percentile → expect within 0.5pp of 69.98%.
4. **3b (Stubs) 16/16 math-sanity:** `--quant.insert-stubs` + w=16 a=16 per_ch + percentile → expect within 0.5pp of 69.98%.
5. **Combined:** both flags on, w=16 a=16 → expect ≤1pp additional drop.
6. **Phase 3 sweep:** both flags on, per_ch + percentile, `w/a = 16/16, 8/8, 4/8, 8/4, 4/4` → results logged to `results/quant_sweep_v2.csv`.

### Phase 3 observed results (2026-04-19)

| w/a/stub | linear | stubs | w_obs | a_obs | stub_obs | top-1 | note |
|---|---|---|---|---|---|---|---|
| 8/8/— (Phase 2 ref) | off | off | per_channel | percentile | — | **59.50** | regression gate target |
| **8/8 (all Phase 3 defaults)** | on | on | per_channel | percentile | percentile | **59.50→57.61** | regression ✓; stubs cost **-1.9pp** (as expected) |
| 16/16 (Phase 2 ref) | off | off | per_channel | percentile | — | **69.98** | 16/16 sanity baseline |
| 16/16 + QuantLinear | on | off | per_channel | percentile | — | 69.95 | 3a math-sanity ✓ (Δ -0.03pp) |
| 16/16 + stubs | off | on | per_channel | percentile | percentile | 69.04 | 3b gate: Δ -0.94pp (compounded percentile clipping at ~29 stubs) |
| 16/16 + linear + stubs | on | on | per_channel | percentile | percentile | 69.04 | combined: classifier adds ≤0.01pp at 16-bit |
| 4/8 Phase 3 | on | on | per_channel | percentile | percentile | 0.16 | 4-bit weights still collapse |
| 8/4 Phase 3 | on | on | per_channel | percentile | percentile | 0.09 | 4-bit activations still collapse |
| 4/4 Phase 3 | on | on | per_channel | percentile | percentile | 0.11 | 4/4 still collapses |

**Takeaways:**
- The 8/8 regression gate reproduces Phase 2 exactly (59.5) — Phase 3 flags default off is backwards-compatible.
- Turning on stubs costs **1.9pp at 8/8** (59.5 → 57.61). The plan predicted a few points of drop as the correct hardware-fidelity cost; observed drop is within that envelope.
- The 16/16 stub gate drops **0.94pp** rather than <0.5pp. This is compounded per-stub percentile clipping, not a simulation leak — the pure QuantLinear gate (no stubs) was within 0.03pp. For a cleaner math-sanity check, use `--quant.stub-observer min_max` which does not clip.
- `QuantLinear` alone contributes essentially nothing over the baseline at both 16-bit and 8-bit — the final classifier is one linear op and its weight range is tame.
- All 4-bit configs (weights or activations) still collapse under the Phase 3 pipeline. The Phase 4 candidates — `MSEObserver`, `HistogramObserver`/KL, AdaRound, bias correction — are the same as they were for Phase 2.

### Key module paths (confirmed)

- `model.conv_1.block.conv` — first Conv2d (skip-list target)
- Every `ConvLayer.block` is `Sequential("conv", "norm", "act")` — `nn.Conv2d` always at `.block.conv`
- Custom `Conv2d(nn.Conv2d)` subclass at `affnet/layers/conv_layer.py:18` — `isinstance(m, nn.Conv2d)` matches correctly
- Depthwise convs (`groups=in_channels`) inside every `InvertedResidual` at `conv_3x3`
- Final classifier: `model.classifier` is `nn.Sequential(global_pool=GlobalPool, [dropout], fc=LinearLayer(1280→1000, channel_first=False))`. `QuantLinear` replaces `.fc`; `StubbedModule` wraps `.global_pool` (and children in `GlobalPool`'s type are caught by the default policy).
- `StubbedModule` wrap points at every `InvertedResidual` (layers 1–2 and every `Block.mlp`), every `Block` inside layer_3/4/5, every `AFFBlock`, and the classifier's `GlobalPool`

---

## Training (minimal — not the workflow focus)

```bash
python main_train.py --log-wandb --common.config-file <config> --common.results-loc <save>
```
Engine: `engine/training_engine.py` (~1600-line Trainer class). Supports DDP, AMP, mixup/cutmix, cosine LR, EMA. Defer to this section only if asked about training.

---

## Environment Setup

- Python 3.8 in conda env `AFFNet`
- **Active Python:** `C:\Users\alona\miniconda3\envs\AFFNet\python.exe` (base env is missing key deps like `pypdf`, `complexPyTorch`)
- Key deps (from `requirements.txt`): `torch==1.13.1`, `torchvision==0.15.2`, `complexPyTorch==0.4`, `torch-dct==0.1.6`, `coremltools==6.2`, `tensorrt==8.5.3.1`
- No `setup.py`/`pyproject.toml` — run in-place from the project root
- Paper PDF lives at `C:\Users\alona\Projects\TokenMixers\AFFNet\docs\adaptive freq filters.pdf` (outside this repo)
