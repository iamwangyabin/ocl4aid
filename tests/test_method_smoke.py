from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


METHODS = [
    "l2p",
    "dualprompt",
    "codaprompt",
    "sprompt",
    "flyprompt",
    "singleprompt",
    "ranpac",
    "slca",
    "hide",
    "hide_lora",
    "hide_adapter",
    "norga",
    "sdlora",
    "mvp",
    "rineside_gauss",
]


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_arrow(path: Path, offset: int) -> None:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [
        (offset + 10, 20, 30),
        (offset + 40, 80, 120),
        (offset + 70, 30, 90),
        (offset + 100, 110, 20),
    ]
    table = pa.Table.from_pydict(
        {
            "image": [_png_bytes(color) for color in colors],
            "label": [0, 1, 0, 1],
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _write_toy_caidbench(root: Path, protocol_path: Path, index_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _write_arrow(root / "Task_A" / "train.arrow", 0)
    _write_arrow(root / "Task_A" / "test.arrow", 5)
    _write_arrow(root / "Task_B" / "train.arrow", 50)
    _write_arrow(root / "Task_B" / "test.arrow", 55)

    rows = []
    for raw_task_id, generator_name, arrow_dir in [
        (10, "Task A", "Task_A"),
        (20, "Task B", "Task_B"),
    ]:
        for split in ["train", "test"]:
            for row_in_batch, label in enumerate([0, 1, 0, 1]):
                rows.append(
                    {
                        "task_id": raw_task_id,
                        "generator_name": generator_name,
                        "raw_generator_name": generator_name,
                        "split": split,
                        "label": label,
                        "arrow_path": f"{arrow_dir}/{split}.arrow",
                        "batch_id": 0,
                        "row_in_batch": row_in_batch,
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), index_path)

    protocol_path.write_text(
        """
index_path: smoke_index.parquet
tasks:
  - id: task_a
    name: Task A
    filter:
      include:
        task_id: 10
  - id: task_b
    name: Task B
    filter:
      include:
        task_id: 20
""".lstrip(),
        encoding="utf-8",
    )


@unittest.skipUnless(
    os.environ.get("OCL4AID_METHOD_SMOKE") == "1",
    "Set OCL4AID_METHOD_SMOKE=1 to run method smoke tests.",
)
class MethodSmokeTests(unittest.TestCase):
    maxDiff = None

    def test_methods_run_toy_protocol(self):
        selected = os.environ.get("OCL4AID_SMOKE_METHODS")
        methods = selected.split(",") if selected else METHODS

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "caidbench"
            protocol_path = tmp_path / "smoke_protocol.yaml"
            index_path = tmp_path / "smoke_index.parquet"
            log_root = tmp_path / "logs"
            _write_toy_caidbench(data_root, protocol_path, index_path)

            failures = {}
            for method in methods:
                cmd = [
                    sys.executable,
                    "main.py",
                    "--config",
                    "configs/framework/caidbench.yaml",
                    "--method",
                    method,
                    "--caidbench_data_dir",
                    str(data_root),
                    "--caidbench_protocol",
                    str(protocol_path),
                    "--caidbench_index_path",
                    str(index_path),
                    "--log_path",
                    str(log_root),
                    "--note",
                    f"smoke_{method}",
                    "--seeds",
                    "1",
                    "--backbone",
                    "vit_tiny_patch16_224",
                    "--no_pretrained",
                    "--no_swanlab",
                    "--no_amp",
                    "--n_worker",
                    "0",
                    "--batchsize",
                    "2",
                    "--base_stage_epochs",
                    "1",
                    "--eval_interval",
                    "0",
                    "--opt_name",
                    "adamw",
                    "--sched_name",
                    "const",
                    "--lr",
                    "0.001",
                    "--online_iter",
                    "1",
                    "--len_prompt",
                    "1",
                    "--len_g_prompt",
                    "1",
                    "--len_e_prompt",
                    "1",
                    "--pos_prompt",
                    "0",
                    "--pos_g_prompt",
                    "0",
                    "--pos_e_prompt",
                    "0",
                    "--e_pool",
                    "4",
                    "--g_pool",
                    "1",
                    "--selection_size",
                    "1",
                    "--key_dim",
                    "192",
                    "--rp_dim",
                    "8",
                    "--ranpac_M",
                    "8",
                    "--adapter_dim",
                    "8",
                    "--sdlora_rank",
                    "2",
                    "--ca_num_per_class",
                    "1",
                    "--ca_steps",
                    "1",
                    "--transforms",
                ]
                result = subprocess.run(
                    cmd,
                    cwd=Path(__file__).resolve().parents[1],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
                metrics_path = log_root / f"smoke_{method}" / "seed_1_ocl_metrics.json"
                if result.returncode != 0 or not metrics_path.is_file():
                    failures[method] = result.stdout[-5000:]

            if failures:
                detail = "\n\n".join(
                    f"--- {method} ---\n{output}" for method, output in failures.items()
                )
                self.fail(f"Method smoke failures:\n{detail}")


if __name__ == "__main__":
    unittest.main()
