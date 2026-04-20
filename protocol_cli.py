"""Command-line utilities for protocol manifest generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_config import GENERATOR_ORDER
from protocol_manifest import build_protocol_from_records, load_records_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="OCL4AID protocol tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_order = subparsers.add_parser("write-generator-order")
    write_order.add_argument("--output", required=True, help="Path to generator_order.json")

    build_manifest = subparsers.add_parser("build-manifest")
    build_manifest.add_argument("--input", required=True, help="Metadata JSONL file")
    build_manifest.add_argument("--output", required=True, help="Output stage_manifest.json file")
    build_manifest.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()
    if args.command == "write-generator-order":
        Path(args.output).write_text(json.dumps(GENERATOR_ORDER, indent=2), encoding="utf-8")
        return

    if args.command == "build-manifest":
        records = load_records_jsonl(args.input)
        protocol = build_protocol_from_records(records, seed=args.seed)
        protocol.write_json(args.output)
        return


if __name__ == "__main__":
    main()
