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

## CLI

Write the fixed generator order:

```bash
python3 -m protocol_cli write-generator-order --output generator_order.json
```

Build a stage manifest from metadata:

```bash
python3 -m protocol_cli build-manifest --input metadata.jsonl --output stage_manifest.json
```

Train with the vendored FlyGCL entrypoint on the protocol dataset:

```bash
python3 main.py \
  --dataset openfake_protocol \
  --protocol_manifest stage_manifest.json \
  --method flyprompt \
  --data_dir /path/to/image/root \
  --note openfake_protocol_run
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Notes

- This repo keeps the continual-learning methods, but drops the generic benchmark dataset support. The only dataset entrypoint is `openfake_protocol`.
- `AIGIBench/ProGAN` is treated as `Stage 0`.
- `OpenFake` generators are the only generators admitted into later training stages.
- `AIGIBench` fake subsets except `ProGAN` are external-only by default.
