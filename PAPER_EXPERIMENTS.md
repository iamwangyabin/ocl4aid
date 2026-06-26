# Paper Experiments

This document is the current experiment plan for the CAIDBenchmark online
continual fake-image detection paper. It replaces the older CAID-10inc,
CAID-AIGC10, and exploratory RINE notes.

## Current Scope

The current paper setting is built around four CAID continual protocols rather
than the older mixed 10inc setting. These four protocol lengths are all main
experiments:

| Protocol | YAML | Role |
| --- | --- | --- |
| Representative10 | `protocol_presets/caidbench/model_appearance_order_protocol_representative10.yaml` | Fast main setting and ablation setting. |
| Representative20 | `protocol_presets/caidbench/model_appearance_order_protocol_representative20.yaml` | Medium main setting. |
| Representative30 | `protocol_presets/caidbench/model_appearance_order_protocol_representative30.yaml` | Large representative main setting. |
| Representative50 | `protocol_presets/caidbench/model_appearance_order_protocol_representative50.yaml` | Full long-horizon main setting. |

`Representative10` stage order:

```text
ProGAN,
Pluralistic,
LaMa,
VQDM,
GLIDE,
Wukong,
Midjourney v5,
RDDM,
Playground,
LaVi-Bridge,
infinity
```

`Representative20` adds 10 more representative generators while preserving CAID
model-appearance order:

```text
ProGAN,
Pluralistic,
StyleGAN3,
LaMa,
VQDM,
GLIDE,
LDM,
StyleGANXL,
Wukong,
DiT,
Midjourney v5,
RDDM,
Kandinsky-3,
Playground,
PixArt-sigma,
LaVi-Bridge,
Janus,
infinity,
GPT-Image-1,
Qwen-Image,
Chroma
```

`Representative30` is the larger representative protocol:

```text
ProGAN,
Pluralistic,
DDIM,
VQGAN,
StyleGAN3,
LaMa,
VQDM,
GLIDE,
LDM,
StyleGANXL,
Wukong,
SD1.5,
SD2.1,
DiT,
Midjourney v5,
SDXL-base,
RDDM,
Kandinsky-3,
Playground,
SiT,
PixArt-sigma,
LaVi-Bridge,
FLUX.1,
Illustrious,
Janus,
SD3.5,
infinity,
GPT-Image-1,
Imagen-4,
Qwen-Image,
Chroma
```

The representative protocols were sampled from the generator pool using stable
held-out test coverage and pooled-detector train/test behavior, then ordered by
CAID model appearance order. `Representative50` keeps the full long-horizon protocol.

## Core Invariants

The learner-visible task remains binary detection:

```text
real = 0
fake = 1
```

Generator names, protocol stage IDs, dataset indices, and benchmark task IDs
define stream/evaluation slices only. They must not be passed into
`online_step`.

Stage 0 is the supervised ProGAN base stage. Online continual learning starts
from stage 1 when `base_stage_epochs > 0`.

## Common Setup

Use the same setup for all rows in one table unless an ablation explicitly
states otherwise:

```text
backbone = vit_base_patch16_224
optimizer = adamw
scheduler = cosine
lr = 0.001
online_iter = 1
batchsize = 16
eval_interval = 20000
seed = 1 for exploratory runs
```

Current fast development setup:

```text
base_stage_epochs = 2
stage_blurry_n = 50
stage_blurry_m = 20
actual leakage = 10%
transforms = []
batch_mask = false
load_base_checkpoint = auto when a matching ProGAN base checkpoint exists
```

Final paper setup should rerun selected rows with:

```text
base_stage_epochs = 10
transforms = [autoaug]
batch_mask = true
seeds = 1 2 3
```

Do not mix fastbase and final-base results in the same table.

## Proposed Method

The current proposed method is:

```text
RIGEv2: Residual Incremental Gaussian Experts v2
```

Implementation:

```text
method = rigev2
model = rigev2
config = configs/methods/rigev2.yaml
```

Final V2 design:

- Train a supervised ProGAN base detector in the raw feature space.
- Select a fixed subset of online features from the trained base head weights.
- Store and replay only the selected feature subset.
- Train per-stage residual low-rank expert heads with replay.
- Route inference using feature-Gaussian expert statistics.
- Keep all training labels binary.

Current default V2 hyperparameters:

```text
rigev2_feature_layers = quartile
rigev2_online_feature_layers = same
rigev2_replay_dim = 1536
rigev2_feature_block_dim = 768
rigev2_head_type = lowrank
rigev2_online_head_type = lowrank
rigev2_rank = 16
rigev2_online_rank = 4
rigev2_eval_mode = feature_gaussian
rigev2_inner_steps = 5
rigev2_replay_window = 8192
rigev2_replay_batch_size = 128
```

The `rigev2_replay_dim=1536` setting is the current final default. A more
compressed 768-dim setting can still be run with `--rigev2_replay_dim 768`, but
it is an ablation row, not the default method.

## Baselines

Main comparison baselines:

```text
l2p
dualprompt
codaprompt
flyprompt
ranpac
```

The proposed row is `rigev2`. `RIGEv1` is not a main baseline row; keep it only
as an internal ablation/storage reference for explaining what V2 changes.

Additional rows when they have complete, validated results:

```text
sprompt
singleprompt
sdlora
mvp
hide
norga
hide_lora
hide_adapter
```

Treat `slca` as invalid until its implementation is verified as a real SLCA
classifier-alignment method for this binary CAID setting. Do not use any method
variant that requires oracle protocol-stage routing at evaluation.

## Metrics

Primary reported metrics:

```text
final_avg_ap
final_avg_auc
final_avg_accuracy
final_avg_f1
final_ap_forgetting
mean_plasticity_ap
```

For online curves, plot:

```text
x-axis = online_sample
y-axis = seen average AP
```

Seen average AP means average AP over generators whose stages have already
appeared by that point. This avoids letting future unseen generators dominate
the curve.

Also keep the final full matrix for per-generator diagnosis:

```text
protocol_matrix.metrics.ap
protocol_matrix.metrics.auc
protocol_matrix.metrics.accuracy
protocol_matrix.metrics.f1
```

## Current Single-Seed Snapshot

These are exploratory `Representative10` single-seed results. They are useful
for method selection and ablation, not final four-protocol paper claims.

| Method / Variant | Protocol | Replay dim | Final Avg AP ↑ | Final Avg Acc ↑ | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| RIGEv1 | Representative10 | 3072 | 0.902430 | 0.826545 | Full feature replay. |
| RIGEv2 | Representative10 | 1536 | 0.905592 | 0.825409 | Current default V2; head-weight selected features. |
| RIGEv2 | Representative10 | 768 | pending | pending | 1/4 feature storage ablation; do not treat as default. |

The current V2 is intended to preserve RIGEv1-level performance while reducing
the replay feature footprint by half.

## Required Experiments

### 1. Four-Protocol Main Comparison

Run the main comparison set under all four protocols:

```text
Representative10 = model_appearance_order_protocol_representative10.yaml
Representative20 = model_appearance_order_protocol_representative20.yaml
Representative30 = model_appearance_order_protocol_representative30.yaml
Representative50 = model_appearance_order_protocol_representative50.yaml
base_stage_epochs = 2 for fast development
stage_blurry_n = 50
stage_blurry_m = 20
actual leakage = 10%
```

Required rows:

```text
l2p, dualprompt, codaprompt, flyprompt, ranpac, rigev2
```

Report final average AP/AUC/accuracy/F1 and AP forgetting. Plot seen average AP
curves for the same rows. `Representative10` should finish first and drive
rapid debugging, but the main paper comparison is the full set of 10/20/30/50
protocols.

### 2. RIGEv2 Ablation

Run on `Representative10` main blurry:

| Variant | Change |
| --- | --- |
| RIGEv2 default | `rigev2_replay_dim=1536` |
| RIGEv2-768 | `--rigev2_replay_dim 768` |
| RIGEv1 | no feature compression |
| no replay | set replay window/batch to 0 if supported |
| hard stream | `stage_blurry_n=100, stage_blurry_m=0` |

Run the V2 ablation on `Representative10` first. If the 1536-dim default remains
the best tradeoff, carry only the default V2 into Representative20,
Representative30, and Representative50. The key question is whether feature
compression keeps current-stage fitting and final seen-generator AP close to
RIGEv1 while reducing stored feature memory.

### 3. Final Paper Reruns

For selected methods only:

```text
base_stage_epochs = 10
transforms = [autoaug]
batch_mask = true
seeds = 1 2 3
```

Use these final reruns for any paper table that claims mean/std.

## Table Templates

### Table 1. Four-Protocol Main Results

Primary table. Fill single-seed values during exploration and `mean ± std`
after final reruns.

| Method | Rep10 AP ↑ | Rep20 AP ↑ | Rep30 AP ↑ | Rep50 AP ↑ | Rep10 Acc ↑ | Rep20 Acc ↑ | Rep30 Acc ↑ | Rep50 Acc ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2P |  |  |  |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |  |  |  |
| FlyPrompt |  |  |  |  |  |  |  |  |
| RanPAC |  |  |  |  |  |  |  |  |
| RIGEv2 | 0.905592 |  |  |  | 0.825409 |  |  |  |

### Table 1b. Four-Protocol Detailed Metrics

Use this table when space allows, or put it in the appendix.

| Protocol | Method | Final Avg AP ↑ | Final Avg AUC ↑ | Final Avg Acc ↑ | Final Avg F1 ↑ | AP Forgetting ↓ | AP Plasticity ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Representative10 | L2P |  |  |  |  |  |  |
| Representative10 | DualPrompt |  |  |  |  |  |  |
| Representative10 | CodaPrompt |  |  |  |  |  |  |
| Representative10 | FlyPrompt |  |  |  |  |  |  |
| Representative10 | RanPAC |  |  |  |  |  |  |
| Representative10 | RIGEv2 | 0.905592 |  | 0.825409 |  |  |  |
| Representative20 | L2P |  |  |  |  |  |  |
| Representative20 | DualPrompt |  |  |  |  |  |  |
| Representative20 | CodaPrompt |  |  |  |  |  |  |
| Representative20 | FlyPrompt |  |  |  |  |  |  |
| Representative20 | RanPAC |  |  |  |  |  |  |
| Representative20 | RIGEv2 |  |  |  |  |  |  |
| Representative30 | L2P |  |  |  |  |  |  |
| Representative30 | DualPrompt |  |  |  |  |  |  |
| Representative30 | CodaPrompt |  |  |  |  |  |  |
| Representative30 | FlyPrompt |  |  |  |  |  |  |
| Representative30 | RanPAC |  |  |  |  |  |  |
| Representative30 | RIGEv2 |  |  |  |  |  |  |
| Representative50 | L2P |  |  |  |  |  |  |
| Representative50 | DualPrompt |  |  |  |  |  |  |
| Representative50 | CodaPrompt |  |  |  |  |  |  |
| Representative50 | FlyPrompt |  |  |  |  |  |  |
| Representative50 | RanPAC |  |  |  |  |  |  |
| Representative50 | RIGEv2 |  |  |  |  |  |  |

### Table 2. Seen Average AP Curve Summary

Fill from `stream_metrics`. Make one curve/table per protocol. The columns
below match Representative10; for Representative20/30/50, extend the same
pattern to their final stage.

| Method | Stage 2 AP ↑ | Stage 4 AP ↑ | Stage 6 AP ↑ | Stage 8 AP ↑ | Stage 10 AP ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2P |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |
| FlyPrompt |  |  |  |  |  |
| RanPAC |  |  |  |  |  |
| RIGEv2 |  |  |  |  |  |

### Table 3. RIGEv2 Storage/Accuracy Ablation

| Variant | Stored feature dim | Replay window | Final Avg AP ↑ | Final Avg Acc ↑ | Relative storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| RIGEv1 | 3072 | 8192 | 0.902430 | 0.826545 | 1.00x |
| RIGEv2-1536 | 1536 | 8192 | 0.905592 | 0.825409 | 0.50x |
| RIGEv2-768 | 768 | 8192 | pending | pending | 0.25x |

### Table 4. Per-Generator Final AP

Fill one row per generator from the final stage of
`protocol_matrix.metrics.ap`.

| Generator | L2P | DualPrompt | CodaPrompt | FlyPrompt | RanPAC | RIGEv2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ProGAN |  |  |  |  |  |  |
| Pluralistic |  |  |  |  |  |  |
| LaMa |  |  |  |  |  |  |
| VQDM |  |  |  |  |  |  |
| GLIDE |  |  |  |  |  |  |
| Wukong |  |  |  |  |  |  |
| Midjourney v5 |  |  |  |  |  |  |
| RDDM |  |  |  |  |  |  |
| Playground |  |  |  |  |  |  |
| LaVi-Bridge |  |  |  |  |  |  |
| infinity |  |  |  |  |  |  |

## Command Templates

For the four main protocols, replace `<protocol_yaml>` with one of:

```text
protocol_presets/caidbench/model_appearance_order_protocol_representative10.yaml
protocol_presets/caidbench/model_appearance_order_protocol_representative20.yaml
protocol_presets/caidbench/model_appearance_order_protocol_representative30.yaml
protocol_presets/caidbench/model_appearance_order_protocol_representative50.yaml
```

Use note/group suffixes such as `rep10`, `rep20`, `rep30`, and `rep50` so
result directories remain separable.

### Local / Generic RIGEv2 Four-Protocol Template

First run, train and save the ProGAN base stage:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method rigev2 \
  --caidbench_data_dir /path/to/CAIDBench \
  --caidbench_protocol <protocol_yaml> \
  --base_stage_epochs 2 \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --transforms \
  --no_batchmask \
  --batchsize 16 \
  --online_iter 1 \
  --eval_interval 20000 \
  --save_base_checkpoint \
  --note rep10_rigev2_s1 \
  --no_swanlab
```

Reuse an existing matching ProGAN base checkpoint:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method rigev2 \
  --caidbench_data_dir /path/to/CAIDBench \
  --caidbench_protocol <protocol_yaml> \
  --base_stage_epochs 2 \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --transforms \
  --no_batchmask \
  --batchsize 16 \
  --online_iter 1 \
  --eval_interval 20000 \
  --load_base_checkpoint auto \
  --note rep10_rigev2_s1_loadbase \
  --no_swanlab
```

### VirtAI RIGEv2 Four-Protocol Template

First run, train and save the ProGAN base stage:

```bash
cd /gemini/code/ocl4aid
export SWANLAB_API_KEY="<redacted>"
ls -lh checkpoints/ViT-B_16.npz
nvidia-smi

python main.py \
  --config configs/framework/caidbench.yaml \
  --method rigev2 \
  --caidbench_data_dir /gemini/data-1/CAIDBench \
  --caidbench_protocol <protocol_yaml> \
  --base_stage_epochs 2 \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --transforms \
  --no_batchmask \
  --batchsize 16 \
  --online_iter 1 \
  --eval_interval 20000 \
  --n_worker 8 \
  --save_base_checkpoint \
  --swanlab \
  --swanlab_project CAIDBench \
  --swanlab_mode cloud \
  --swanlab_group <protocol_tag>-mainblurry \
  --swanlab_experiment_name <protocol_tag>-rigev2-s1 \
  --swanlab_tags <protocol_tag> mainblurry rigev2 \
  --log_path /gemini/output/ocl4aid_logs \
  --note <protocol_tag>_rigev2_s1
```

Reuse an existing matching base checkpoint:

```bash
cd /gemini/code/ocl4aid
export SWANLAB_API_KEY="<redacted>"
ls -lh checkpoints/ViT-B_16.npz
nvidia-smi

python main.py \
  --config configs/framework/caidbench.yaml \
  --method rigev2 \
  --caidbench_data_dir /gemini/data-1/CAIDBench \
  --caidbench_protocol <protocol_yaml> \
  --base_stage_epochs 2 \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --transforms \
  --no_batchmask \
  --batchsize 16 \
  --online_iter 1 \
  --eval_interval 20000 \
  --n_worker 8 \
  --load_base_checkpoint auto \
  --swanlab \
  --swanlab_project CAIDBench \
  --swanlab_mode cloud \
  --swanlab_group <protocol_tag>-mainblurry \
  --swanlab_experiment_name <protocol_tag>-rigev2-s1-loadbase \
  --swanlab_tags <protocol_tag> mainblurry rigev2 loadbase \
  --log_path /gemini/output/ocl4aid_logs \
  --note <protocol_tag>_rigev2_s1_loadbase
```

### Baseline Template

Replace `<method>` with `l2p`, `dualprompt`, `codaprompt`, `flyprompt`,
or `ranpac`.

```bash
python main.py \
  --config configs/framework/caidbench.yaml \
  --method <method> \
  --caidbench_data_dir /gemini/data-1/CAIDBench \
  --caidbench_protocol <protocol_yaml> \
  --base_stage_epochs 2 \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --transforms \
  --no_batchmask \
  --batchsize 16 \
  --online_iter 1 \
  --eval_interval 20000 \
  --n_worker 8 \
  --save_base_checkpoint \
  --swanlab \
  --swanlab_project CAIDBench \
  --swanlab_mode cloud \
  --swanlab_group <protocol_tag>-mainblurry \
  --swanlab_experiment_name <protocol_tag>-<method>-s1 \
  --swanlab_tags <protocol_tag> mainblurry <method> \
  --log_path /gemini/output/ocl4aid_logs \
  --note <protocol_tag>_<method>_s1
```

Method-specific overrides:

```text
codaprompt: add --e_pool 110
dualprompt: add --lr 0.005 --e_pool 110 --len_g_prompt 20 --len_e_prompt 50 --pos_e_prompt 2 3 4 5 6 7 8 9
ranpac: add --ranpac_M 4096
rigev2-768 ablation: add --rigev2_replay_dim 768
```

## Result Files

Each completed seed writes:

```text
<log_path>/<method>/seed_<seed>_ocl_metrics.json
```

Use:

```text
final_summary
metrics
protocol_matrix
stage_metrics
stream_metrics
```

For paper tables, extract from `final_summary`. For curves, extract from
`stream_metrics`. For per-generator diagnosis, extract from `protocol_matrix`
at the final stage.

Do not commit datasets, checkpoints, SwanLab logs, run logs, or archived result
bundles.
