# ocl4aid

OpenFake protocol continual fake detection codebase.

## What Is Implemented

- Fixed `generator_order.json` for the 29-stage curriculum.
- Protocol modules at repo root:
  - `protocol_config.py`
  - `protocol_manifest.py`
  - `protocol_metrics.py`
  - `protocol_cli.py`
- `protocol_manifest.py` to:
  - load metadata rows from JSONL
  - assign fake samples into contiguous `70/20/10` blurry windows
  - assign non-overlapping real slices with per-stage `real == fake_total`
  - build balanced internal and external test slices
  - write `stage_manifest.json`
- Vendored `FlyGCL` training stack under:
  - `main.py`
  - `configuration/`
  - `methods/`
  - `models/`
  - `utils/`
  - `datasets/`
- A protocol-specific dataset and sampler:
  - `datasets/OpenFakeProtocol.py`
  - explicit stage sampling in `utils/onlinesampler.py`
- Protocol-aware evaluation in `methods/_trainer.py`
  - balanced internal slice accuracy
  - balanced external subset accuracy
  - `Average Accuracy / Forgetting / Plasticity` JSON output
- The protocol dataset exposes class ids and explicit stage membership through the manifest-backed `OpenFakeProtocol` loader.
- `protocol_metrics.py` computes:
  - `avg_accuracy_by_stage`
  - `forgetting_by_stage`
  - `plasticity_by_stage`
  - `external_accuracy_by_stage`
- `tests/test_protocol.py` with toy protocol validation.

## Metadata Input Format

Build inputs as JSONL with at least:

```json
{"record_id":"...", "path":"...", "source_dataset":"openfake", "split":"train", "binary_label":"fake", "generator_name":"Stable Diffusion 1.5"}
```

Required fields:

- `record_id`
- `path`
- `source_dataset`
- `split`
- `binary_label`

Optional fields:

- `generator_name`
- `subset_name`
- `release_date`
- `parquet_path`
- `parquet_row_index`
- `parquet_image_column`

## CLI

Write the fixed generator order:

```bash
python3 -m protocol_cli write-generator-order --output generator_order.json
```

Build a stage manifest from metadata:

```bash
python3 -m protocol_cli build-manifest --input metadata.jsonl --output stage_manifest.json
```

OpenFake can be pulled from Hugging Face (`ComplexDataLab/OpenFake`)
automatically when `--protocol_manifest` is omitted. That path is intended for
small smoke runs. For the full OpenFake v2 protocol, build one of the fixed-size
manifests below so the sampled rows and continual stages are fixed.

If you want to control where files live, pass a path explicitly:

- `--openfake_hf_cache_dir /path/to/hf_cache` uses a non-default Hugging Face
  dataset cache.
- `--data_dir /path/to/ocl4aid_openfake_cache` stores/uses the auto-prepared
  OCL4AID smoke-run files there.
- `--protocol_manifest /path/to/stage_manifest.json` uses a prebuilt protocol
  manifest. `--data_dir` is optional when manifest records already contain
  absolute parquet paths.

Build a compact OpenFake-only manifest manually only when you want explicit
control over the cached files:

```bash
python3 tools/export_openfake_subset.py \
  --output-dir data/openfake_smoke \
  --generators "Stable Diffusion 1.5" "Stable Diffusion 2.1"

python3 -m protocol_cli build-manifest \
  --input data/openfake_smoke/metadata.jsonl \
  --output data/openfake_smoke/stage_manifest.json \
  --openfake-only
```

### OpenFake v2 fixed-size protocol presets

For the current OpenFake v2 release, use the fixed-sampling manifest builder when
the full `core/train` split is too large for online continual learning. It
uses the model release/count CSV bundled in `protocol_presets/` and automatically
finds the newest `ComplexDataLab/OpenFake` snapshot from the Hugging Face cache.
It deterministically selects up to `K` fake rows per
training generator, selects the same number of real training rows, and includes
full evaluation splits:

- `core/validation` as in-domain / seen-generator evaluation
- `core/test` as OOD unseen-generator evaluation
- `reddit/test` as Wild evaluation

Preset configs are provided under `protocol_presets/`:

- `openfake_v2_k500.json`
- `openfake_v2_k1000.json`
- `openfake_v2_k5000.json`

Example on a server that already downloaded `ComplexDataLab/OpenFake`:

```bash
python3 tools/export_openfake_v2_protocol.py --config protocol_presets/openfake_v2_k1000.json
```

The command writes:

- `metadata.jsonl`
- `generator_order.json`
- `stage_manifest.json`
- `selection_summary.json`

Selected records point back to their parquet file and row index; no images are
exported or copied. Training reads images directly from the local Hugging Face
snapshot.

Train from the exported subset:

```bash
python3 main.py \
  --dataset openfake_protocol \
  --method flyprompt \
  --protocol_manifest data/openfake_v2_k1000/stage_manifest.json \
  --note openfake_v2_k1000 \
  --protocol_external_eval_period 0
```

`--protocol_external_eval_period 0` evaluates full OOD/Wild slices only at the
final stage. Use `1` to evaluate them after every stage, or a larger integer to
evaluate every N stages.

If the Hugging Face cache is not in the default location, pass `--snapshot-root`
or set `HF_HOME` / `HF_HUB_CACHE`.

Train with the vendored FlyGCL entrypoint on the protocol dataset. This command
auto-prepares OpenFake from Hugging Face using the default cache:

```bash
python3 main.py \
  --dataset openfake_protocol \
  --method flyprompt \
  --note openfake_protocol_run
```

Use `--protocol_manifest` when you want to train from a prebuilt protocol
manifest.

SwanLab tracking is enabled by default for training, evaluation, and protocol
metrics:

```bash
python3 main.py \
  --dataset openfake_protocol \
  --method flyprompt \
  --note openfake_protocol_run \
  --swanlab_project ocl4aid \
  --swanlab_workspace your_workspace
```

The SwanLab project defaults to `ocl4aid`. The experiment name defaults to
`<note_or_method>_<YYYYmmdd_HHMMSS>`, so the example above is recorded like
`openfake_protocol_run_20260514_153000` unless `--swanlab_experiment_name` is
provided.
The SwanLab run logs `train/*`, `test/*`, `task/*`, `summary/*`, and
`protocol/*` metrics from the main process only. Use `--swanlab_mode local`
for local-only runs, or `--no_swanlab` to disable tracking.

## Tests

```bash
python3 -m unittest discover -s tests
```

## Notes

- This repo keeps the continual-learning methods, but drops the generic benchmark dataset support. The only dataset entrypoint is `openfake_protocol`.
- `AIGIBench/ProGAN` is treated as `Stage 0`.
- `OpenFake` generators are the only generators admitted into later training stages.
- `AIGIBench` fake subsets except `ProGAN` are external-only by default.
