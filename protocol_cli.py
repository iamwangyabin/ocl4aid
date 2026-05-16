"""Command-line utilities for protocol manifest generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_config import GENERATOR_ORDER
from protocol_manifest import (
    build_protocol_from_records,
    load_generator_order_json,
    load_records_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCL4AID protocol tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_order = subparsers.add_parser("write-generator-order")
    write_order.add_argument("--output", required=True, help="Path to generator_order.json")

    build_manifest = subparsers.add_parser("build-manifest")
    build_manifest.add_argument("--input", required=True, help="Metadata JSONL file")
    build_manifest.add_argument("--output", required=True, help="Output stage_manifest.json file")
    build_manifest.add_argument("--seed", type=int, default=13)
    build_manifest.add_argument(
        "--generator-order",
        default=None,
        help="Optional generator_order.json for a custom stage curriculum.",
    )
    build_manifest.add_argument(
        "--openfake-only",
        action="store_true",
        help="Build a compact OpenFake-only protocol without AIGIBench/ProGAN stages.",
    )
    build_manifest.add_argument(
        "--include-external-tests",
        action="store_true",
        default=False,
        help="Include external test slices even when --openfake-only is set.",
    )
    build_manifest.add_argument(
        "--external-source-dataset",
        action="append",
        default=None,
        help="Source dataset to treat as external test data. Can be repeated.",
    )

    args = parser.parse_args()
    if args.command == "write-generator-order":
        Path(args.output).write_text(json.dumps(GENERATOR_ORDER, indent=2), encoding="utf-8")
        return

    if args.command == "build-manifest":
        records = load_records_jsonl(args.input)
        generator_order = (
            load_generator_order_json(args.generator_order)
            if args.generator_order is not None
            else None
        )
        protocol = build_protocol_from_records(
            records,
            seed=args.seed,
            openfake_only=args.openfake_only,
            generator_order=generator_order,
            include_external_tests=(
                True if args.include_external_tests else None
            ),
            external_source_datasets=args.external_source_dataset,
        )
        protocol.write_json(args.output)
        return


if __name__ == "__main__":
    main()
