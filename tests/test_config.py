from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ConfigParserTests(unittest.TestCase):
    def _base_parser(self):
        fake_methods = types.ModuleType("methods")
        fake_methods.METHODS = {"l2p": object()}
        config_path = Path(__file__).resolve().parents[1] / "configuration" / "config.py"
        spec = importlib.util.spec_from_file_location("_test_configuration_config", config_path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"methods": fake_methods}):
            spec.loader.exec_module(module)
        return module.base_parser

    def test_base_checkpoint_cli_flags_parse(self):
        base_parser = self._base_parser()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            checkpoint_dir = Path(tmp) / "base_ckpts"
            config_path.write_text(
                """
data:
  root: /tmp/CAIDBench
tracking:
  swanlab: false
""".lstrip(),
                encoding="utf-8",
            )
            argv = [
                "prog",
                "--config",
                str(config_path),
                "--method",
                "l2p",
                "--save_base_checkpoint",
                "--base_checkpoint_dir",
                str(checkpoint_dir),
                "--load_base_checkpoint",
                "auto",
                "--base_checkpoint_only",
                "--no_swanlab",
            ]

            with patch.object(sys, "argv", argv):
                args = base_parser()

        self.assertTrue(args.save_base_checkpoint)
        self.assertEqual(args.base_checkpoint_dir, str(checkpoint_dir))
        self.assertEqual(args.load_base_checkpoint, "auto")
        self.assertTrue(args.base_checkpoint_only)

    def test_base_checkpoint_yaml_defaults_parse(self):
        base_parser = self._base_parser()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
data:
  root: /tmp/CAIDBench
train:
  save_base_checkpoint: true
  base_checkpoint_dir: /tmp/base_ckpts
  load_base_checkpoint: auto
  base_checkpoint_only: true
tracking:
  swanlab: false
""".lstrip(),
                encoding="utf-8",
            )
            argv = [
                "prog",
                "--config",
                str(config_path),
                "--method",
                "l2p",
            ]

            with patch.object(sys, "argv", argv):
                args = base_parser()

        self.assertTrue(args.save_base_checkpoint)
        self.assertEqual(args.base_checkpoint_dir, "/tmp/base_ckpts")
        self.assertEqual(args.load_base_checkpoint, "auto")
        self.assertTrue(args.base_checkpoint_only)

    def test_face_bbox_path_yaml_and_cli_parse(self):
        base_parser = self._base_parser()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
data:
  root: /tmp/CAIDBench
  face_bbox_path: /tmp/from_yaml.parquet
tracking:
  swanlab: false
""".lstrip(),
                encoding="utf-8",
            )
            argv = [
                "prog",
                "--config",
                str(config_path),
                "--method",
                "l2p",
                "--caidbench_face_bbox_path",
                "/tmp/from_cli.parquet",
            ]

            with patch.object(sys, "argv", argv):
                args = base_parser()

        self.assertEqual(args.caidbench_face_bbox_path, "/tmp/from_cli.parquet")

    def test_method_yaml_defaults_and_extra_cli_overrides_parse(self):
        base_parser = self._base_parser()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                """
data:
  root: /tmp/CAIDBench
tracking:
  swanlab: false
""".lstrip(),
                encoding="utf-8",
            )
            argv = [
                "prog",
                "--config",
                str(config_path),
                "--method",
                "l2p",
                "--len_e_prompt",
                "7",
                "--no_batchwise_prompt_selection",
            ]

            with patch.object(sys, "argv", argv):
                args = base_parser()

        self.assertEqual(args.len_e_prompt, 7)
        self.assertEqual(args.e_pool, 30)
        self.assertFalse(args._batchwise_selection)


if __name__ == "__main__":
    unittest.main()
