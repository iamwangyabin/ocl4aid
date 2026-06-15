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

## Train

```bash
python3 main.py \
  --caidbench_data_dir /path/to/CAIDBench \
  --method flyprompt \
  --caidbench_label_mode generator \
  --num_epochs 1 \
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
results/logs/caidbench_protocol/<note>/
```

The main output is `seed_<seed>_ocl_metrics.json`, containing per-stage
accuracy, average accuracy, forgetting, and plasticity.

## Tests

```bash
python3 -m unittest discover -s tests
```
