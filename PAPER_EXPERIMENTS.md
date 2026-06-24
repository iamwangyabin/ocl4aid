# Paper Experiments

This document tracks the experiments needed for the CAIDBenchmark online
continual deepfake detection paper.

## Common Setup

All method comparisons should use the same framework configuration unless a
table explicitly states otherwise.

```text
base_stage_epochs = 10
backbone = vit_base_patch16_224
optimizer = adamw
scheduler = cosine
lr = 0.001
online_iter = 1
batchsize = 16
eval_interval = 20000
```

Stage 0 is the supervised base stage. In the default protocol this is ProGAN.
With `base_stage_epochs=10`, every method first trains a supervised base detector
on ProGAN for 10 epochs, then online continual learning starts from stage 1.
Do not compare methods with different base-stage budgets.

The preferred long-horizon protocol is `CAID-50`, defined by:

```text
protocol_presets/caidbench/model_appearance_order_protocol_50.yaml
```

`CAID-50` keeps the original time order, uses 50 generator stages, and requires
each generator to have a balanced test split of 2000 images: 1000 real and 1000
fake. This keeps final sample-level and generator-level test summaries aligned.

The short-horizon protocol is `CAID-10inc`, defined by:

```text
protocol_presets/caidbench/model_appearance_order_protocol_10inc.yaml
```

`CAID-10inc` is a faster diagnostic setting with one supervised base generator
and 10 online incremental generator stages. It should use the same common
training setup as `CAID-50`, including `base_stage_epochs=10`, backbone,
optimizer, batch size, online update budget, and evaluation interval. Use it to
debug method behavior and to provide a compact short-horizon comparison; do not
mix `CAID-10inc` numbers with the main `CAID-50` table.

The base stage checkpoint can be saved once per method/seed and reused across
stream settings. Use `--save_base_checkpoint --base_checkpoint_only` to
precompute the base, then use `--load_base_checkpoint auto` for hard, mild,
main, and strong blurry runs with the same method, backbone, protocol order,
seed, and `base_stage_epochs`. If different stream queues use different
`log_path` roots, set a shared `--base_checkpoint_dir` for all of them.

The learner supervision remains binary throughout all experiments:

```text
real = 0
fake = 1
```

Generator stages define the stream and evaluation slices only. Dataset indices,
generator names, protocol stage IDs, and benchmark task IDs must not be passed
into `online_step`.

## VirtAI Runtime Notes

For the VirtAI SSH job environment, use the mounted project directories rather
than `/root` for persistent files:

```text
code:      /gemini/code        ($GEMINI_CODE)
data:      /gemini/data-1      ($GEMINI_DATA_IN1)
data:      /gemini/data-2      ($GEMINI_DATA_IN2)
data:      /gemini/data-3      ($GEMINI_DATA_IN3)
pretrain:  /gemini/pretrain    ($GEMINI_PRETRAIN)
pretrain2: /gemini/pretrain2   ($GEMINI_PRETRAIN2)
pretrain3: /gemini/pretrain3   ($GEMINI_PRETRAIN3)
output:    /gemini/output      ($GEMINI_DATA_OUT, offline training)
```

Place this repository under `/gemini/code/ocl4aid`. Container-local paths such
as `/root` are writable but temporary and may be lost when the container is
restarted. Read datasets from `/gemini/data-*` and pretrained weights from
`/gemini/pretrain*`.

One checked VirtAI job exposes eight RTX 3090 devices through
`/proc/driver/nvidia/gpus/*/information`, although `nvidia-smi` may print
`SMI N/A`/`Driver Version: N/A` in the SSH shell. Before launching training,
verify CUDA from Python after installing PyTorch:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## Stream Settings

The main benchmark should be the blurry stream. The hard stream is a control
setting, not the primary benchmark.

Temporal blur is applied only to online stages when the base stage is enabled.
The effective adjacent-stage leakage ratio is:

```text
actual_leakage = (100 - stage_blurry_n) * stage_blurry_m / 10000
```

Run these stream settings:

| Setting | `stage_blurry_n` | `stage_blurry_m` | Effective leakage |
| --- | ---: | ---: | ---: |
| Hard control | 100 | 0 | 0% |
| Mild blurry | 50 | 10 | 5% |
| Main blurry | 50 | 20 | 10% |
| Strong blurry | 50 | 40 | 20% |

Use `Main blurry` as the default paper setting.

## Methods

Required core methods:

- `flyprompt`
- `l2p`
- `dualprompt`
- `codaprompt`
- `mvp`
- `ranpac`

Additional methods for complete paper tables when compute allows:

- `slca`
- `sprompt`
- `singleprompt`
- `sdlora`
- `hide`
- `norga`
- `hide_lora`
- `hide_adapter`

All methods in the same table should use the same seed, backbone, batch size,
online update budget, base-stage budget, and evaluation interval. For
exploratory remote runs, use a single seed unless a final paper table explicitly
requires multi-seed averaging.

## Required Experiments

### 1. Main Blurry Method Comparison

Run all methods under:

```text
base_stage_epochs = 10
stage_blurry_n = 50
stage_blurry_m = 20
actual leakage = 10%
```

This is the main paper table. During exploration, report final-stage summary
metrics for the selected seed. Run multi-seed averaging only for final paper
tables after the method set is fixed.

Minimum methods:

```text
flyprompt, l2p, dualprompt, codaprompt, mvp, ranpac
```

Complete methods:

```text
flyprompt, l2p, dualprompt, codaprompt, mvp, ranpac, slca, sprompt,
singleprompt, sdlora, hide, norga, hide_lora, hide_adapter
```

### 2. Hard Control Method Comparison

Run the same methods under:

```text
base_stage_epochs = 10
stage_blurry_n = 100
stage_blurry_m = 0
actual leakage = 0%
```

This is a control table showing performance under clean generator-stage
boundaries.

### 3. Blurry Strength Comparison

Compare stream difficulty across:

```text
Hard control   n=100, m=0   leakage=0%
Mild blurry    n=50,  m=10  leakage=5%
Main blurry    n=50,  m=20  leakage=10%
Strong blurry  n=50,  m=40  leakage=20%
```

If compute is limited, run only the core methods. If compute allows, run all
methods.

### 4. Online Curves

Use `stream_metrics` from `seed_<seed>_ocl_metrics.json` to plot online
performance over stream samples.

Required curves for the main blurry setting:

```text
flyprompt, l2p, dualprompt, codaprompt, mvp, ranpac
```

Recommended axes:

```text
x-axis: online samples seen
y-axis: average accuracy / f1 / ap / auc
```

The main figure can show accuracy and AUC. F1 and AP can go to the appendix.

### 5. Short-Horizon 10-Increment Comparison

Run the same core methods on `CAID-10inc`:

```text
protocol_presets/caidbench/model_appearance_order_protocol_10inc.yaml
```

This setting has one supervised base generator and 10 online incremental
generators. It is mainly for rapid method debugging, ablation checks, and a
compact short-horizon result table. Keep the same common setup and stream
setting as the corresponding `CAID-50` experiment. For example, the short main
blurry run should still use:

```text
base_stage_epochs = 10
stage_blurry_n = 50
stage_blurry_m = 20
actual leakage = 10%
```

Report final-stage average metrics and online curves separately from the
long-horizon `CAID-50` results.

### 6. Final Summary Tables

For each method and stream setting, report final-stage values from
`seed_<seed>_ocl_metrics.json`.

Primary metrics:

```text
avg accuracy
avg auc
forgetting
plasticity
```

Secondary metrics:

```text
avg f1
avg ap
```

### 7. Per-Generator Results

For the main blurry setting, export final per-generator metrics for all
generators in the selected protocol. For `CAID-50`, this means all 50 generator
stages.

Use this for appendix tables or heatmaps:

```text
generator x method: accuracy / auc / forgetting
```

This analysis should identify which generators are hardest and which earlier
generators suffer the most forgetting.

## Paper Table Templates

Use these tables as the final paper experiment skeleton. Fill values from the
final stage of `seed_<seed>_ocl_metrics.json` unless the table explicitly says
otherwise. Use `mean ± std` only after multi-seed runs are available; for
single-seed exploration, fill a single value.

### Table 1. Main Results on CAID-50 Main Blurry

Protocol: `CAID-50`; stream: `stage_blurry_n=50, stage_blurry_m=20`; base:
ProGAN, 10 epochs.

| Method | Final Avg Acc ↑ | Final Avg AUC ↑ | Final Avg AP ↑ | Final Avg F1 ↑ | AUC Forgetting ↓ | AUC Plasticity ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlyPrompt |  |  |  |  |  |  |
| L2P |  |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |  |
| MVP |  |  |  |  |  |  |
| RanPAC |  |  |  |  |  |  |
| SLCA |  |  |  |  |  |  |
| SPrompt |  |  |  |  |  |  |
| SinglePrompt |  |  |  |  |  |  |
| SD-LoRA |  |  |  |  |  |  |
| HiDe |  |  |  |  |  |  |
| NoRGa |  |  |  |  |  |  |
| HiDe-LoRA |  |  |  |  |  |  |
| HiDe-Adapter |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |

### Table 2. Hard-Control Results on CAID-50

Protocol: `CAID-50`; stream: `stage_blurry_n=100, stage_blurry_m=0`; base:
ProGAN, 10 epochs. This table isolates clean generator-stage boundaries.

| Method | Final Avg Acc ↑ | Final Avg AUC ↑ | Final Avg AP ↑ | Final Avg F1 ↑ | AUC Forgetting ↓ | AUC Plasticity ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlyPrompt |  |  |  |  |  |  |
| L2P |  |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |  |
| MVP |  |  |  |  |  |  |
| RanPAC |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |

### Table 3. Blurry Strength Ablation on CAID-50

Use the same method set and base checkpoint policy for all rows. If compute is
limited, fill this table with core methods only.

| Method | Hard AUC ↑ | Mild AUC ↑ | Main AUC ↑ | Strong AUC ↑ | Hard Fgt ↓ | Mild Fgt ↓ | Main Fgt ↓ | Strong Fgt ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FlyPrompt |  |  |  |  |  |  |  |  |
| L2P |  |  |  |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |  |  |  |
| MVP |  |  |  |  |  |  |  |  |
| RanPAC |  |  |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |  |  |

### Table 4. Short-Horizon CAID-10inc Results

Protocol: `CAID-10inc`; stream should match the corresponding CAID-50 setting,
usually main blurry.

| Method | Final Avg Acc ↑ | Final Avg AUC ↑ | Final Avg AP ↑ | Final Avg F1 ↑ | AUC Forgetting ↓ | AUC Plasticity ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FlyPrompt |  |  |  |  |  |  |
| L2P |  |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |  |
| MVP |  |  |  |  |  |  |
| RanPAC |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |

### Table 5. Ablation Study for the Proposed Method

Use CAID-50 main blurry unless stated otherwise.

| Variant | Final Avg Acc ↑ | Final Avg AUC ↑ | Final Avg AP ↑ | Final Avg F1 ↑ | AUC Forgetting ↓ | AUC Plasticity ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full method |  |  |  |  |  |  |
| w/o face crop |  |  |  |  |  |  |
| w/o base checkpoint |  |  |  |  |  |  |
| w/o online calibration |  |  |  |  |  |  |
| hard stream only |  |  |  |  |  |  |
| 10inc short protocol |  |  |  |  |  |  |

### Appendix Table A1. Final Per-Generator Results

Fill one row per generator from the final-stage matrix. Use this table for each
important method, or convert it into a generator-by-method heatmap.

| Generator | Acc ↑ | AUC ↑ | AP ↑ | F1 ↑ | Forgetting AUC ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| ProGAN |  |  |  |  |  |
| DeepFakes |  |  |  |  |  |
| BigGAN |  |  |  |  |  |
| ... |  |  |  |  |  |
| Z-Image |  |  |  |  |  |

### Appendix Table A2. Forward and Backward Transfer

Compute these from the full protocol matrix.

| Method | Forward AUC ↑ | Forward AP ↑ | Backward AUC ↑ | Backward AP ↑ | Final Avg AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlyPrompt |  |  |  |  |  |
| L2P |  |  |  |  |  |
| DualPrompt |  |  |  |  |  |
| CodaPrompt |  |  |  |  |  |
| MVP |  |  |  |  |  |
| RanPAC |  |  |  |  |  |
| Ours |  |  |  |  |  |

## Active Execution Plan

Last updated: 2026-06-20 CST.

### CAID-10inc Result Audit

Last consolidated on 2026-06-24 CST. The local source of truth is
`outputs/caid10inc_results/summary.tsv`, with the full archived bundle at
`outputs/caid10inc_results.tar.gz`.

Complete CAID-10inc main-blurry runs currently available for a fair main table:

| Method | Archived run | Notes |
| --- | --- | --- |
| FlyPrompt | `flyprompt_s1` | Complete final-stage result. |
| L2P | `l2p_s1` | Complete final-stage result. |
| DualPrompt | `dualprompt_bigprompt_s1` | Tuned prompt capacity; complete final-stage result. |
| CodaPrompt | `codaprompt_s1` | Complete final-stage result. |
| RanPAC | `ranpac_m4096_s1` | Complete final-stage result copied from `4090-2`; random projection dimension 4096. |
| SD-LoRA | `sdlora_s1` | Complete final-stage result. |
| RINE-Residual | `rine_residual_base_s1` | Non-oracle result; currently the only RINE row suitable for a main table. |

CAID-10inc rows still missing from the full template:

| Method | Status |
| --- | --- |
| MVP | No complete archived `seed_1_ocl_metrics.json` result. |
| SLCA | Current `slca_s1` run is invalid: AP/AUC/F1 matrices are unchanged across all protocol stages, training loss stays near 0.693, and the code path is a plain full-ViT trainer rather than a real SLCA/classifier-alignment implementation. The result has been moved to `outputs/caid10inc_invalid_results/slca_s1`. |
| SPrompt | Previously summarized, but the source `seed_1_ocl_metrics.json` is not currently present in `outputs/caid10inc_results/`; recover the raw result before using it in source-of-truth tables or curves. |
| SinglePrompt | The available `caid10inc_singleprompt_s1(1).tgz` contains only logs/SwanLab files and no final metrics JSON. |
| NoRGa | No complete archived `seed_1_ocl_metrics.json` result. |
| HiDe | Existing `hide_s1` result used the original class-to-task fallback, which is invalid for binary CAID labels. Rerun after enabling the learned RPFC generator-stage router. |
| HiDe-LoRA | Existing `hide_lora_s1` result used the original class-to-task fallback, which is invalid for binary CAID labels. Rerun after enabling the learned RPFC generator-stage router. |
| HiDe-Adapter | Existing `hide_adapter_s1` result used the original class-to-task fallback, which is invalid for binary CAID labels. Rerun after enabling the learned RPFC generator-stage router. |

Rows with `rine_taskoracle_*` or task-oracle-like routing are diagnostic only.
They must not be used as main paper results because the evaluation route uses
protocol-stage information that is unavailable in a task-agnostic deployment.
These diagnostic RINE directories have been removed from the local
`outputs/caid10inc_results/` archive; keep only `rine_residual_base_s1` there.

Larger paper tables remain missing: all CAID-50 main-blurry results, all
CAID-50 hard-control results, blurry-strength ablations, proposed-method
ablations, and multi-seed mean/std runs. Current 10inc numbers are single-seed
exploratory results.

### RINE 10inc Status

Last checked on `4090-2` on 2026-06-24. Task-oracle routing is not an
acceptable protocol assumption for paper results: it uses the evaluation slice's
protocol stage to select the expert head. RINE code now removes `task_oracle`
and all `task_oracle_*` eval modes; valid runs must use task-agnostic head
selection or aggregation. Current RINE-Residual code keeps only the baseline
non-oracle eval modes `max_fake` and `max_confidence`. The exploratory
`calibrated_mean`, `shared_online`, `base + residual`, prototype, memory-kNN,
and online-router code paths were removed after negative checks.

Invalidated task-oracle diagnostics found under `/home/yabin/ocl4aid/run_logs/`:

| Run | Protocol stage | Final avg AP | Final avg AUC | Notes |
| --- | ---: | ---: | ---: | --- |
| `caid10inc_rine_residual_independent_taskoracle_linearhead_augbase100k_balprior_s1` | 10 | 0.8278 | 0.8295 | Invalid as a main result because expert selection uses protocol stage ID at evaluation. |
| `caid10inc_rine_residual_independent_taskoracle_augbase100k_lr3e4_step1_fd01_s1_20260623` | 10 | 0.8258 | 0.8270 | Invalid as a main result; useful only as a diagnostic that lower LR helps BigGAN but hurts SD1.5/SDXL. |
| `caid10inc_rine_residual_independent_baseinit_augbase100k_balprior_s1_20260623` | 10 | 0.8181 | 0.8246 | Invalid as a main result if run with task-oracle routing. |
| `caid10inc_rine_residual_independent_baseinit_taskoracle_rank16_step2_loadbase_s1_20260623` | 10 | 0.7853 | 0.7933 | Invalid as a main result. |
| `caid10inc_rine_residual_taskoracle_linearhead_degrade_j05_ds05_blur03_s1_20260623` | 10 | 0.7781 | 0.7864 | Invalid as a main result. |
| `caid10inc_rine_residual_independent_taskoracle_linearhead_step2_s1_20260623` | 10 | 0.7727 | 0.7780 | Invalid as a main result. |

Rows using reusable base checkpoints should be treated as fast direction checks,
not final paper numbers. Current rescue runs load the augmented 100k 10inc base
checkpoint and use strict `n=100,m=0` online exposure.

Additional audit on `4090-2` confirms the 10inc stream itself is balanced and
does not explain the RINE collapse. The training split has balanced real/fake
labels in every online stage: ProGAN `360059/360059`, DeepFakes `5000/5000`,
BigGAN `1506/1506`, StyleGAN2/DDIM/LDM `10000/10000`, and SD1.5/Midjourney
v5/SDXL-base/FLUX.1/GPT-Image-1 `5000/5000`. Every test generator slice is
also `1000/1000`. The stream sampler interleaves labels within each stage, so
there is no one-class online exposure causing the head collapse.

The main negative finding is stronger: even invalid task-oracle diagnostics are
not close to 0.9 final avg AP, and the same generators stay weak when the
correct expert is selected. For example, the strongest invalid task-oracle
linear-head run reaches only final avg AP `0.8278`; its final StyleGAN2 AP is
`0.4837`, DeepFakes AP `0.6709`, SD1.5 AP `0.6534`, and SDXL-base AP `0.7219`.
This means head routing/aggregation alone is not a credible path to 0.9 unless
the per-generator online heads themselves become much stronger.

Exploratory aggregation branches were checked only as single-seed diagnostics
and then deleted from the codebase. None of those branches read dataset indices,
generator names, protocol stage IDs, or evaluation labels during training;
`online_step` remained image + binary-label only.

No valid 10inc final result is currently close to 0.9 AP. The strongest
complete 10inc result that does not use task-oracle routing is
`caid10inc_rine_residual_base1_s1_20260623` with final avg AP 0.6198. The best
valid partial RINE direction so far is still only an early-stage diagnostic:
`max_fake` reaches avg AP 0.8234 at stage 2, then falls to 0.6623 at stage 3.

An `online_router` single-seed 10inc validation run was launched and stopped on
`4090-2` on 2026-06-24 because it underperformed at the known StyleGAN2
collapse point:

```text
run = caid10inc_rine_residual_onlinerouter_augbase100k_balprior_s1_20260624
eval_mode = online_router
stage 1 avg AP = 0.8212
stage 2 avg AP = 0.7930
stage 3 matrix avg AP = 0.6515
stage 3 StyleGAN2 AP = 0.4771
verdict = valid but negative; online-router code removed from current path
```

The best deleted non-oracle aggregation diagnostic was:

```text
run = caid10inc_rine_residual_calibmean_addbase_v2_augbase100k_balprior_s1_20260624
eval_mode = calibrated_mean
add_base = true
residual_scale = 0.2
stage 2 matrix avg AP = 0.8409
stage 3 matrix avg AP = 0.7581
stage 3 StyleGAN2 AP = 0.5537
stage 10 matrix avg AP = 0.6691
stage 10 matrix avg AUC = 0.6766
weak final slices = DeepFakes 0.5531 AP, StyleGAN2 0.5083 AP, SD1.5 0.4735 AP, SDXL-base 0.5601 AP, FLUX.1 0.5413 AP, GPT-Image-1 0.5731 AP
verdict = valid, improves early/mid-stage stability, but not a 0.9 final-AP route
```

Deleted shared online binary-head diagnostics:

```text
run = caid10inc_rine_residual_sharedonline_replay10k_augbase100k_balprior_s1_20260624
eval_mode = shared_online
head = random linear
replay = 10k online feature replay, replay batch 128
add_base = true
stage 3 matrix avg AP = 0.7418
stage 3 StyleGAN2 AP = 0.5487
stage 5 stream avg AP = 0.7544
verdict = valid but weaker than calibrated_mean+add_base; stopped early
```

```text
run = caid10inc_rine_residual_sharedonline_lowrankinit_basereplay5k_replay20k_s1_20260624
eval_mode = shared_online
head = lowrank initialized from base head
replay = 5k base-stage train feature seed + 20k online feature replay
add_base = false
stage 1 matrix avg AP = 0.4723
stage 1 ProGAN AP = 0.3072, AUC = 0.0044
verdict = valid but failed immediately; stopped early
```

Negative aggregation checks:

| Run | Last checked stage | Avg AP | Baseline at same stage | Notes |
| --- | ---: | ---: | ---: | --- |
| `caid10inc_rine_residual_maxfake_augbase100k_balprior_s1_20260623` | 3 | 0.6623 | n/a | Valid `max_fake`; stage 2 reached 0.8234, but StyleGAN2 current AP fell to 0.4831 at stage 3. |
| `caid10inc_rine_residual_maxconf_augbase100k_balprior_s1_20260623` | 3 | 0.6634 | 0.6623 | Valid `max_confidence`; stage 1 improved to 0.8129 but stage 3 still collapsed. |
| `caid10inc_rine_residual_protorouter_augbase100k_balprior_s1_20260623b` | 3 | 0.7166 | n/a | Pure feature-prototype router; task-agnostic, but DeepFakes and stage-3 average AP dropped. |
| `caid10inc_rine_residual_protodet_augbase100k_balprior_s1_20260623` | 2 | 0.6411 | 0.8234 | Direct binary prototype detector; stage 1 AP 0.7200 and stage 2 AP 0.6411, stopped early. |
| `caid10inc_rine_residual_memoryknn20k_k50_augbase100k_balprior_s1_20260623` | 1 | 0.7150 | 0.7924 | Online binary feature-memory kNN; DeepFakes current AP 0.4317, stopped early. |
| `caid10inc_rineside_gauss_stats100k_logmeanexp_s1_20260623` | 3 | 0.6899 | 0.6623 | Pure online Gaussian/prototypical stats with 100k base samples; valid but still weak, StyleGAN2 AP 0.5697. |
| `caid10inc_rine_residual_independent_calibmax_augbase100k_balprior_s1_20260623` | 3 | 0.6629 | n/a | Centered `calibrated_max`; task-agnostic, stopped early because it was clearly weak. |
| `caid10inc_rine_residual_independent_centeredmax_rank16_step2_loadbase_s1_20260623` | 2 | 0.7167 | 0.7300 | Centered `calibrated_max` under the older reusable-base setup; stopped early. |
| `caid10inc_rine_residual_independent_calibmax_rank16_step2_loadbase_s1_20260623` | 4 | about 0.62 | 0.7313 | Quality-weighted calibrated max; stopped early. |

Remote runs should use a single seed by default. Multi-seed runs are reserved
for final paper tables after the method and stream setting are selected. Current
RINE rescue runs explicitly override the framework YAML defaults:

```text
base_stage_epochs = 2
load_base_checkpoint = run_logs/base_checkpoints_aug/base_rine_residual_vit_base_patch16_224_model_appearance_order_protocol_10inc_seed1_stage0_epochs2.pt
backbone = vit_base_patch16_224
optimizer = adamw
scheduler = cosine
lr = 0.001
online_iter = 1
batchsize = 16
eval_interval = 20000
seeds = 1
method = rine_residual
stage_blurry_n = 100
stage_blurry_m = 0
```

### CAID-10inc DualPrompt Big-Prompt Diagnostic

Launched on `A6000` on 2026-06-23 CST. This is an exploratory rescue run for
DualPrompt base-stage underfitting on the short `CAID-10inc` main-blurry
protocol. It is not part of the common paper setup because it uses a different
base-stage budget, larger prompt capacity, no AutoAugment, and a higher
learning rate.

Run identity:

```text
method = dualprompt
protocol = protocol_presets/caidbench/model_appearance_order_protocol_10inc.yaml
stream = main blurry, n=50, m=20, leakage=10%
seed = 1
machine = A6000
pid_at_launch = 2660435
swanlab = https://swanlab.cn/@iamwan/ocl4aid/runs/3c3f0vq5q9mw83yt1r1r8
train_log = /home/home/yabin/ocl4aid/run_logs/caid10inc_dualprompt_s1_bigprompt_savebase_e110_le50_lr5e3_b2/seed_1_train.log
launch_log = /home/home/yabin/ocl4aid/run_logs/caid10inc_dualprompt_s1_bigprompt_savebase_e110_le50_lr5e3_b2_20260623_230005.log
```

Confirmed configuration from the launch log:

```text
base_stage_epochs = 2
save_base_checkpoint = true
base_checkpoint_dir = /home/home/yabin/ocl4aid/run_logs/base_checkpoints
batchsize = 16
online_iter = 1
eval_interval = 20000
n_worker = 24
lr = 0.005
transforms = []
no_batchmask = true
e_pool = 110
len_g_prompt = 20
len_e_prompt = 50
pos_g_prompt = [0, 1]
pos_e_prompt = [2, 3, 4, 5, 6, 7, 8, 9]
total_parameters = 119707394
learnable_parameters = 33908738
```

Expected base checkpoint path after base stage finishes:

```text
/home/home/yabin/ocl4aid/run_logs/base_checkpoints/base_dualprompt_vit_base_patch16_224_model_appearance_order_protocol_10inc_seed1_stage0_epochs2.pt
```

Exact launch command, with the SwanLab API key intentionally redacted:

```bash
cd /home/home/yabin/ocl4aid

export SWANLAB_API_KEY="<redacted>"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

/home/home/yabin/miniconda3/envs/cl/bin/python main.py \
  --config configs/framework/caidbench.yaml \
  --method dualprompt \
  --caidbench_data_dir /home/home/yabin/CAIDBench \
  --caidbench_protocol protocol_presets/caidbench/model_appearance_order_protocol_10inc.yaml \
  --base_stage_epochs 2 \
  --save_base_checkpoint \
  --base_checkpoint_dir /home/home/yabin/ocl4aid/run_logs/base_checkpoints \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --online_iter 1 \
  --batchsize 16 \
  --eval_interval 20000 \
  --n_worker 24 \
  --swanlab \
  --swanlab_project ocl4aid \
  --swanlab_mode cloud \
  --swanlab_group caid10inc-mainblurry \
  --swanlab_experiment_name caid10inc-dualprompt-s1-bigprompt-savebase-e110-le50-lr5e3-b2 \
  --swanlab_tags caid10inc mainblurry dualprompt bigprompt savebase e110 le50 lr5e3 b2 noautoaug \
  --log_path /home/home/yabin/ocl4aid/run_logs \
  --note caid10inc_dualprompt_s1_bigprompt_savebase_e110_le50_lr5e3_b2 \
  --no_batchmask \
  --e_pool 110 \
  --len_g_prompt 20 \
  --len_e_prompt 50 \
  --pos_g_prompt 0 1 \
  --pos_e_prompt 2 3 4 5 6 7 8 9 \
  --lr 0.005 \
  --transforms
```

Current A6000 SPrompt run configuration:

```text
method = sprompt
stream = main blurry
stage_blurry_n = 50
stage_blurry_m = 20
base_stage_epochs = 10
backbone = vit_base_patch16_224
pretraining = ImageNet-21k ViT-B/16, ViT-B_16.npz
seed = 1
batchsize = 16
online_iter = 1
eval_interval = 20000
n_worker = 8
base_checkpoint_args = --save_base_checkpoint --base_checkpoint_dir /home/home/yabin/ocl4aid/run_logs/base_checkpoints
```

The older CAID experiment logs were moved into per-machine
`run_logs/_archive_before_base_reuse_<timestamp>/` directories before launching
this run. The A6000-to-4090-1 CAIDBench rsync transfer was not stopped and is
not part of this experiment plan.

Current machine assignment and status:

| Machine | Stream setting | Plan id | Remote commit | Data root | Status |
| --- | --- | --- | --- | --- | --- |
| `4090-2` | Main blurry, `n=50,m=20`, leakage 10% | `mbrpfix0620` | `78fcf03` | `/home/yabin/CAIDBench` | running single-seed `ranpac` |
| `A6000` | Main blurry, `n=50,m=20`, leakage 10% | `sprompt_mainblurry_base10_s1_20260620` | latest `main` | `/home/home/yabin/CAIDBench` | planned single-seed `sprompt` run |

Launcher script:

```text
scripts/launch_caid_experiment_queue.sh
```

Legacy base-5 main blurry logs and base checkpoints on `4090-2`:

```text
/home/yabin/ocl4aid/run_logs/caid_mainblurry_baseckpt_core_s1to3_20260618/
/home/yabin/ocl4aid/run_logs/caid_mainblurry_<method>_base5_s1-2-3_78fcf03/
/home/yabin/ocl4aid/run_logs/base_checkpoints/
```

A6000 SPrompt main blurry logs and base checkpoints:

```text
/home/home/yabin/ocl4aid/run_logs/sprompt_mainblurry_base10_s1_20260620/
/home/home/yabin/ocl4aid/run_logs/caid_mainblurry_sprompt_base10_s1_<commit>/
/home/home/yabin/ocl4aid/run_logs/base_checkpoints/
```

Reusable base checkpoint filename pattern:

```text
base_<method>_vit_base_patch16_224_model_appearance_order_protocol_seed<seed>_stage0_epochs10.pt
```

Future same-machine stream runs can reuse the saved base with:

```bash
--load_base_checkpoint auto --base_checkpoint_dir <machine>/run_logs/base_checkpoints
```

Monitoring commands:

```bash
ssh 4090-2 "tail -n 80 /home/yabin/ocl4aid/run_logs/caid_mainblurry_baseckpt_core_s1to3_20260618/launcher.log"
ssh A6000 "tail -n 80 /home/home/yabin/ocl4aid/run_logs/sprompt_mainblurry_base10_s1_20260620/launcher.log"
ssh 4090-2 "find /home/yabin/ocl4aid/run_logs/base_checkpoints -maxdepth 1 -type f -name '*.pt' | sort"
ssh A6000 "find /home/home/yabin/ocl4aid/run_logs/base_checkpoints -maxdepth 1 -type f -name '*.pt' | sort"
```

After these queues finish, run the remaining planned stream-strength settings
for the same core methods:

```text
Mild blurry    n=50, m=10, leakage=5%
Strong blurry  n=50, m=40, leakage=20%
```

Then expand to the additional methods and seeds only after the core-method
results are stable.

## Run Count

Core-method development run:

```text
4 stream settings x 6 core methods x 3 seeds = 72 runs
```

Core-method final run:

```text
4 stream settings x 6 core methods x 5 seeds = 120 runs
```

Complete final run:

```text
4 stream settings x 14 methods x 5 seeds = 280 runs
```

Recommended execution order:

1. Run `Main blurry` and `Hard control` with core methods and 3 seeds.
2. If results are stable, expand `Main blurry` to all methods and 5 seeds.
3. Run `Mild blurry` and `Strong blurry` for core methods.
4. Expand `Mild blurry` and `Strong blurry` to all methods only if compute
   allows.
5. Generate online curves and per-generator appendix results from completed
   runs.

## Example Commands

Precompute reusable base:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --base_stage_epochs 10 \
  --save_base_checkpoint \
  --base_checkpoint_only \
  --no_swanlab
```

Main blurry:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --base_stage_epochs 10 \
  --load_base_checkpoint auto \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --note flyprompt_base10_blurry10 \
  --no_swanlab
```

Hard control:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --base_stage_epochs 10 \
  --load_base_checkpoint auto \
  --stage_blurry_n 100 \
  --stage_blurry_m 0 \
  --note flyprompt_base10_hard \
  --no_swanlab
```

Final paper seeds:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --seeds 1 2 3 4 5 \
  --base_stage_epochs 10 \
  --load_base_checkpoint auto \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --note flyprompt_base10_blurry10_s5 \
  --no_swanlab
```
