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

## Quantization (upcoming — not yet implemented)

**Goal:** Post-Training Quantization (PTQ) for on-device deployment (TensorRT / CoreML / ONNX target).

**Planned strategy — mixed precision:**
- Conv / Linear / BN layers → **INT8** via `torch.quantization` (fbgemm for x86 CPU, qnnpack for ARM mobile); fuse Conv+BN+Act first
- AFNO2D frequency layers → **FP16** (autocast + cast `w1/w2/b1/b2` to half). `rfft2`/`irfft2` and complex einsum have no standard INT8 kernels, so FP16 is the pragmatic compromise
- Calibration: reuse `create_eval_loader()` with N batches through the prepared model
- Benchmark: reuse `Evaluator` to compare FP32 vs mixed-precision on accuracy, model size, latency

**Planned layout** (to be created):
```
quantization/
├── ptq_pipeline.py       # Orchestrator
├── model_prep.py         # fuse_modules + qconfig + QuantStubs
├── afno_fp16.py          # AFNO2D FP16 wrapper
├── calibration.py        # Run N batches through prepared model
└── export.py             # ONNX / CoreML / TensorRT export
main_quant.py             # Entry point, mirrors main_eval.py
```

Key challenge: `ConvLayer` stores `Conv2d + Norm + Act` as sub-modules — need to read `affnet/layers/conv_layer.py` carefully for exact attribute paths before calling `torch.quantization.fuse_modules`.

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
