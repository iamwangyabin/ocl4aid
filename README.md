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
before the online continual stream starts. The default framework config uses
10 epochs. Set it to 0 only when stage 0 should be included in the online
continual stream instead.

`--batchsize` is the online exposure batch size. Training is intentionally
single-process and single-GPU so one update always corresponds to exactly that
many newly exposed stream samples. Machines may have multiple visible GPUs, but
the runner uses only the first visible device; select another device with
`CUDA_VISIBLE_DEVICES`. Distributed launchers fail fast because sharding the
stream changes online ordering and batch semantics.

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
  --config configs/framework/caidbench_50_hard.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
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
  --config configs/framework/caidbench_50_mainblurry.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
  --load_base_checkpoint auto \
  --no_swanlab
```

## Configuration

Common framework settings live in `configs/framework/caidbench.yaml`. Method
settings live under `configs/methods/`. The loader reads the small shared
fallback file `configs/methods/common.yaml`, then overlays
`configs/methods/<method>.yaml`.

Checked-in framework presets cover the current paper and diagnostic launch
settings, so protocol, base-stage, stream-blur, batch, evaluation, and tracking
defaults do not need to be repeated on the command line:

```text
configs/framework/caidbench_50_hard.yaml
configs/framework/caidbench_50_mainblurry.yaml
configs/framework/caidbench_aigc10_mainblurry.yaml
configs/framework/caidbench_aigc10_mainblurry_fastbase.yaml
configs/framework/caidbench_10inc_mainblurry.yaml
configs/framework/caidbench_10inc_mainblurry_fastbase.yaml
```

The AIGC10 presets are the preferred short-horizon diagnostic setting: ProGAN
is the supervised base stage, followed by 10 modern AIGC-era generators in CAID
appearance order. The older CAID-10inc presets are retained for reproducing
existing mixed classic/recent short-protocol runs. The AIGC10 and CAID-10inc
presets include per-method experiment overrides currently used for
`codaprompt`, `dualprompt`, and `ranpac`.

`configuration/config.py` only declares framework-level CLI flags. Method
hyperparameters are injected from the method YAML, and can still be overridden
from the command line with the same key name, for example
`--len_prompt 10` or `--rigev1_inner_steps 5`. Do not add a new parser
entry for every method-specific option.

The default framework optimizer is AdamW with `CosineAnnealingLR`; the default
learning rate is `0.001` and can be overridden with `--lr`.

Every registered method should have a corresponding method YAML, even when it
only documents that the method uses common defaults. Keep framework/run
settings such as data paths, seeds, batch size, and evaluation interval in the
framework YAML; keep method architecture and method-specific algorithm settings
in the method YAML.

## Train

```bash
python3 main.py \
  --config configs/framework/caidbench_50_hard.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
  --no_swanlab
```

Override the protocol or index only when needed:

```bash
python3 main.py \
  --caidbench_data_dir /path/to/CAIDBench \
  --caidbench_protocol protocol_presets/caidbench/model_appearance_order_protocol.yaml \
  --caidbench_index_path protocol_presets/caidbench/continual_index.parquet
```

For fast method iteration, use the short ProGAN-plus-AIGC10 protocol:

```bash
python3 main.py \
  --config configs/framework/caidbench_aigc10_mainblurry_fastbase.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
  --log_path run_logs \
  --no_swanlab
```

## Logging

SwanLab is enabled by default. Disable it with `--no_swanlab`, or configure:

```bash
python3 main.py \
  --config configs/framework/caidbench_aigc10_mainblurry.yaml \
  --method flyprompt \
  --caidbench_data_dir /path/to/CAIDBench \
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

The summary also reports average performance, forgetting, plasticity, BWT, and
FWT for each metric. In blurry streams, these use the sampler's actual
per-generator exposure window rather than the nominal stage number: FWT is
measured only from an evaluation before the first exposed sample, while
forgetting/BWT start after the last exposed sample. Generators completed in the
final stream bucket are excluded from forgetting/BWT because there is no later
stage in which forgetting can be observed; the JSON includes the valid-term
counts for each aggregate.
These exposure-aware definitions are identified by `metrics_schema_version: 2`;
do not pool them with older summaries that used the previous denominator and
nominal blurry-stage boundaries.

CodaPrompt now expands a non-divisible prompt pool to complete per-task slices.
Legacy CodaPrompt base checkpoints whose prompt-pool shape changes are rejected
with an explicit regeneration error rather than being loaded ambiguously.

## Tests

```bash
python3 -m unittest discover -s tests
```
