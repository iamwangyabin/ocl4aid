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

## Training Labels

Training is always binary:

- `real = 0`
- `fake = 1`

The protocol still keeps generator/stage metadata internally so it can build the
stream order and report per-generator evaluation metrics. That metadata is not
used as class supervision and is not passed into `online_step`.

## Online Setting

The protocol treats each generator stage as a framework-level continual task,
following the FlyGCL-style separation between stream structure and learner
supervision. Stage 0 is treated as the supervised base stage. In the default
protocol this stage is ProGAN, and the trainer runs it as conventional
supervised base training before online continual learning begins at stage 1.
The trainer may call method task-boundary hooks, but `online_step` receives
only mini-batches of images and binary labels. Dataset indices, generator
names, and protocol stage IDs are not passed into `online_step`.

`task_num` is set to the number of protocol generator stages so prompt/expert
methods can allocate one slot per framework task. The class supervision exposed
to the learner remains binary.

`--online_iter` controls how many optimizer updates are run when a mini-batch
arrives.

`--base_stage_epochs` controls how many supervised epochs are run on stage 0
before the online continual stream starts. The default is 1. Set it to 0 only
when stage 0 should be included in the online continual stream instead.

`--batchsize` is the global online exposure batch size, not a per-GPU batch
size. In distributed training the trainer splits it evenly across ranks before
building each local dataloader, so one synchronized online update still
corresponds to the requested global number of newly exposed stream samples. The
value must be divisible by `world_size`; otherwise training exits instead of
silently changing the online setting.

Temporal stage blur is controlled by `--stage_blurry_n`/`--stage_blurry_m` (or
`--n`/`--m`). With the default base stage enabled, stage 0 remains a clean base
stage and temporal blur is applied only to online stages from stage 1 onward.

Periodic online evaluation uses framework-only stream offsets: by default the
trainer evaluates every 20000 training samples using the full test slices for
the generators that have appeared in the stream so far. These evaluations log
binary deepfake detection metrics and are kept in
`seed_<seed>_ocl_metrics.json`.

## Base Checkpoints

The supervised base stage can be saved once and reused by later runs with the
same method, backbone, protocol order, seed, and `--base_stage_epochs` value.
This avoids rerunning stage 0 when comparing stream settings.

Precompute and save the base stage:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --base_stage_epochs 5 \
  --save_base_checkpoint \
  --base_checkpoint_only \
  --no_swanlab
```

The automatic save path is:

```text
<log_path>/base_checkpoints/base_<method>_<backbone>_<protocol>_seed<seed>_stage0_epochs<epochs>.pt
```

When different queues use different `--log_path` roots, pass the same
`--base_checkpoint_dir` to both the precompute command and the online runs.

Reuse it in an online run:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --base_stage_epochs 5 \
  --load_base_checkpoint auto \
  --stage_blurry_n 50 \
  --stage_blurry_m 20 \
  --no_swanlab
```

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

Each seed writes:

- `seed_<seed>_train.log`: full stdout-style training and protocol-evaluation log
- `seed_<seed>_ocl_metrics.json`: structured stage, stream, and summary metrics

The main output is `seed_<seed>_ocl_metrics.json`, containing per-generator and
per-stage binary deepfake detection metrics:

- `accuracy`
- `f1`
- `ap`
- `auc`

The summary also reports average performance, forgetting, and plasticity for
each metric.

## Tests

```bash
python3 -m unittest discover -s tests
```
