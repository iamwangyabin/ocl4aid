# ocl4aid

Online continual learning for CAIDBenchmark.

This repository keeps the continual-learning methods and training loop for the
CAIDBenchmark protocol.

## Dataset

Training expects a CAIDBenchmark Arrow package on disk and the paired protocol
index under:

```text
protocol_presets/caidbench/
  model_appearance_order_protocol.yaml
  continual_index.parquet
```

The default protocol is the model-appearance chronological order. The actual
image Arrow files are not bundled; pass their root with `--caidbench_data_dir`.
The index selects rows by `arrow_path`, `batch_id`, and `row_in_batch` relative
to that root.

## Label Modes

`--caidbench_label_mode generator` is the default:

- `real = 0`
- fake samples from online stage `k` use class `k + 1`
- evaluation folds predictions back to binary real/fake accuracy

`--caidbench_label_mode binary` trains directly on CAID labels:

- `real = 0`
- `fake = 1`

## Online Setting

The protocol YAML defines the continual process: active stage order, the base
stage, and every later online stage. Stage 0 is treated as the base session.
`--base_epochs` only controls how many passes are used for that first active
protocol stage; every later online stage is observed once. `--online_iter`
controls how many optimizer updates are run when a mini-batch arrives.

FlyPrompt, DualPrompt, and MVP keep their task-free internal prompt-session
schedule, but it is active only during the online phase. By default the number
of internal sessions is inferred from the protocol stage count, with the base
session occupying slot 0.

Periodic online evaluation follows a FlyPrompt-style stream checkpoint: by
default the trainer evaluates every 20000 online training samples using the full
test slices for the generators seen so far. These stream evaluations are logged
separately from the stage-boundary metrics in `seed_<seed>_ocl_metrics.json`.

## Configuration

Common framework settings live in `configs/framework/caidbench.yaml`. Method
settings live under `configs/methods/<method>.yaml` when a method needs its own
hyperparameters. CLI flags still override YAML values.

## Train

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --no_swanlab
```

Override the protocol or index only when needed:

```bash
python3 main.py \
  --caidbench_data_dir /path/to/CAIDBench \
  --caidbench_protocol protocol_presets/caidbench/model_appearance_order_protocol.yaml \
  --caidbench_index_path protocol_presets/caidbench/continual_index.parquet
```

## Logging

SwanLab is enabled by default. Disable it with `--no_swanlab`, or configure:

```bash
python3 main.py \
  --caidbench_data_dir /path/to/CAIDBench \
  --swanlab_project ocl4aid \
  --swanlab_workspace your_workspace
```

Metrics are written under:

```text
run_logs/<note>/
```

The main output is `seed_<seed>_ocl_metrics.json`, containing per-stage
accuracy, average accuracy, forgetting, and plasticity. After the first active
stage completes, `seed_<seed>_after_base_task.pt` is saved in the same run
directory for reuse.

## Tests

```bash
python3 -m unittest discover -s tests
```
