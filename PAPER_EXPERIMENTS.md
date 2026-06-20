# Paper Experiments

This document tracks the experiments needed for the CAIDBenchmark online
continual deepfake detection paper.

## Common Setup

All method comparisons should use the same framework configuration unless a
table explicitly states otherwise.

```text
base_stage_epochs = 10
backbone = vit_base_patch16_224
online_iter = 1
batchsize = 16
eval_interval = 20000
```

Stage 0 is the supervised base stage. In the default protocol this is ProGAN.
With `base_stage_epochs=10`, every method first trains a supervised base detector
on ProGAN for 10 epochs, then online continual learning starts from stage 1.
Do not compare methods with different base-stage budgets.

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

### 5. Final Summary Tables

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

### 6. Per-Generator Results

For the main blurry setting, export final per-generator metrics for all 90
protocol generator stages.

Use this for appendix tables or heatmaps:

```text
generator x method: accuracy / auc / forgetting
```

This analysis should identify which generators are hardest and which earlier
generators suffer the most forgetting.

## Active Execution Plan

Last updated: 2026-06-20 CST.

Remote runs should use a single seed by default. Multi-seed runs are reserved
for final paper tables after the method and stream setting are selected. The
current remote run explicitly overrides the framework YAML defaults so that the
paper common setup is used:

```text
base_stage_epochs = 10
backbone = vit_base_patch16_224
online_iter = 1
batchsize = 16
eval_interval = 20000
seeds = 1
methods = flyprompt, l2p, dualprompt, codaprompt, mvp, ranpac
base_checkpoint_args = --save_base_checkpoint --base_checkpoint_dir <machine>/run_logs/base_checkpoints
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
