# AGENTS.md

## Project Overview

This repository implements online continual learning for CAIDBenchmark. The
training loop is in `methods/_trainer.py`, the CLI/config entry point is
`main.py` and `configuration/config.py`, and CAIDBenchmark protocol loading is
implemented in `datasets/CAIDBenchmarkProtocol.py`.

Training is binary deepfake detection:

- `real = 0`
- `fake = 1`

Generator and protocol-stage metadata define framework-level continual tasks
and evaluation slices, but it must not become class supervision. `online_step`
must still receive only images and binary labels.

## Key Paths

- `main.py`: CLI entry point.
- `configuration/config.py`: argparse and YAML default loading.
- `configs/framework/`: framework-level configs.
- `configs/methods/`: method-specific hyperparameters.
- `methods/`: trainer classes and online update logic.
- `models/`: model definitions selected by method name.
- `datasets/CAIDBenchmarkProtocol.py`: Arrow/index-backed protocol dataset.
- `utils/onlinesampler.py`: protocol stream and stage/task samplers.
- `protocol_metrics.py`: binary detection and online continual metrics.
- `tests/`: unittest-based protocol and metric tests.

## Setup Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all tests:

```bash
python3 -m unittest discover -s tests
```

Run a typical training job:

```bash
python3 main.py \
  --config configs/framework/caidbench.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
  --no_swanlab
```

## Core Invariants

- Keep learner supervision binary even though the framework tracks generator
  tasks.
- Do not pass dataset indices, generator names, protocol stage IDs, or benchmark
  task IDs into `online_step`.
- The protocol YAML controls generator task order and evaluation slices.
- `online_step(images, labels, None)` is intentional.
- Stage 0 is the supervised base stage. In the default protocol this is ProGAN.
  Online continual learning starts from stage 1 unless `base_stage_epochs` is
  set to 0.
- When the base stage is enabled, temporal stage blur must start after stage 0
  and must not leak samples into or out of the supervised base stage.
- `batchsize` is the online exposure batch size. The runner is intentionally
  single-process and single-GPU; distributed launchers must fail fast rather
  than shard the online stream or reinterpret the batch size.
- `task_num` should match the number of protocol generator stages so
  prompt/expert methods can allocate per-task slots. This must not change the
  binary class labels exposed to training.
- Binary metrics should use hard binary predictions for accuracy/F1 and fake
  confidence scores for AP/AUC.
- AP/AUC may be `None` when the evaluated slice does not contain enough class
  variation.
- For exploratory remote training runs, use a single seed unless the user
  explicitly asks for multi-seed averaging. Do not queue multiple seeds by
  default.

## Adding or Updating Methods

When adding a new method:

- Add the trainer under `methods/`.
- Register it in `methods/__init__.py`.
- Add the model under `models/` when needed.
- Register the model in `models/__init__.py`.
- Add `configs/methods/<method>.yaml` if the method has method-specific
  defaults.
- Add new CLI defaults in `configuration/config.py` only when they are actually
  needed by the method.
- Implement protocol evaluation support in `_protocol_eval_logits` if the
  method needs custom inference behavior.

Follow existing trainer patterns: call `add_new_class(labels)`, map labels
through `self.exposed_classes`, apply `self.train_transform`, use `self.mask`
for exposed-class masking, and step the scheduler through `update_schedule()`.

## Testing Guidance

For dataset/protocol changes, run:

```bash
python3 -m unittest tests.test_caidbench_protocol
```

For metric changes, run:

```bash
python3 -m unittest tests.test_protocol_metrics
```

Before finishing any non-trivial change, run:

```bash
python3 -m unittest discover -s tests
```

Training smoke tests require a local CAIDBenchmark Arrow package and may require
GPU resources. If the data or GPU is unavailable, state that explicitly.

## Data and Artifacts

Do not commit local datasets, checkpoints, generated experiment logs, SwanLab
outputs, or large run artifacts. Keep generated outputs under ignored locations
such as `run_logs/`, `outputs/`, `results/`, or `data/`.

Do not introduce new hard-coded personal absolute paths. Prefer CLI flags or
YAML settings that can be overridden.

## Code Style

Prefer the existing project style over broad refactors. Keep changes scoped to
the relevant method, model, dataset, metric, or config module. Use concise
comments only when they explain non-obvious protocol or evaluation behavior.
