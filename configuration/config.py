import argparse
from pathlib import Path

from methods import METHODS


DEFAULT_FRAMEWORK_CONFIG = "configs/framework/caidbench.yaml"
_METHOD_CLI_ALIASES = {
    "batchwise_prompt_selection": "_batchwise_selection",
    "diversed_prompt_selection": "_diversed_selection",
}


def _load_yaml(path):
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.is_file():
        return {}
    import yaml
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return payload


def _get_nested(payload, *keys):
    cur = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _framework_defaults(payload):
    mapping = {
        ("data", "root"): "caidbench_data_dir",
        ("data", "protocol"): "caidbench_protocol",
        ("data", "index"): "caidbench_index_path",
        ("data", "image_column"): "caidbench_image_column",
        ("data", "face_bbox_path"): "caidbench_face_bbox_path",
        ("run", "seeds"): "seeds",
        ("run", "note"): "note",
        ("run", "log_dir"): "log_path",
        ("model", "backbone"): "backbone",
        ("model", "pretrained"): "pretrained",
        ("train", "optimizer"): "opt_name",
        ("train", "scheduler"): "sched_name",
        ("train", "amp"): "use_amp",
        ("train", "workers"): "n_worker",
        ("train", "batch_size"): "batchsize",
        ("train", "lr"): "lr",
        ("train", "online_iter"): "online_iter",
        ("train", "base_stage_epochs"): "base_stage_epochs",
        ("train", "save_base_checkpoint"): "save_base_checkpoint",
        ("train", "base_checkpoint_dir"): "base_checkpoint_dir",
        ("train", "load_base_checkpoint"): "load_base_checkpoint",
        ("train", "base_checkpoint_only"): "base_checkpoint_only",
        ("train", "stage_blurry_n"): "stage_blurry_n",
        ("train", "stage_blurry_m"): "stage_blurry_m",
        ("train", "transforms"): "transforms",
        ("train", "topk"): "topk",
        ("eval", "interval"): "eval_interval",
        ("tracking", "swanlab"): "use_swanlab",
    }
    defaults = {}
    for path, dest in mapping.items():
        value = _get_nested(payload, *path)
        if value is not None:
            defaults[dest] = value
    batch_mask = _get_nested(payload, "train", "batch_mask")
    if batch_mask is not None:
        defaults["no_batchmask"] = not bool(batch_mask)
    return defaults


def _method_defaults(method):
    defaults = _load_yaml(Path("configs/methods") / "common.yaml")
    defaults.update(_load_yaml(Path("configs/methods") / f"{method}.yaml"))
    return defaults


def _parse_cli_value(value):
    import yaml
    return yaml.safe_load(value)


def _coerce_extra_cli_value(key, values, defaults):
    default = defaults.get(key)
    if not values:
        return True

    if isinstance(default, bool):
        if len(values) != 1:
            raise ValueError(f"--{key} expects a boolean value")
        parsed = _parse_cli_value(values[0])
        if isinstance(parsed, bool):
            return parsed
        if isinstance(parsed, str):
            text = parsed.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"--{key} expects a boolean value, got {values[0]!r}")

    if isinstance(default, list):
        if len(values) == 1:
            parsed = _parse_cli_value(values[0])
            return parsed if isinstance(parsed, list) else [parsed]
        return [_parse_cli_value(value) for value in values]

    if len(values) == 1:
        parsed = _parse_cli_value(values[0])
        if default is None or isinstance(parsed, type(default)):
            return parsed
        try:
            return type(default)(parsed)
        except (TypeError, ValueError):
            return parsed

    return [_parse_cli_value(value) for value in values]


def _parse_extra_cli_overrides(tokens, defaults):
    overrides = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument {token!r}")

        key = token[2:].replace("-", "_")
        idx += 1
        if key.startswith("no_"):
            positive_key = _METHOD_CLI_ALIASES.get(key[3:], key[3:])
            overrides[positive_key] = False
            continue
        key = _METHOD_CLI_ALIASES.get(key, key)

        values = []
        while idx < len(tokens) and not tokens[idx].startswith("--"):
            values.append(tokens[idx])
            idx += 1
        overrides[key] = _coerce_extra_cli_value(key, values, defaults)
    return overrides


def base_parser():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_FRAMEWORK_CONFIG,
                            help="Framework-level YAML config.")
    pre_parser.add_argument("--method", type=str, default=None, choices=METHODS.keys(),
                            help="Method name")
    pre_args, _ = pre_parser.parse_known_args()

    framework_config = _load_yaml(pre_args.config)
    defaults = _framework_defaults(framework_config)
    method = pre_args.method or "l2p"
    defaults.update(_method_defaults(method))
    defaults["method"] = method

    parser = argparse.ArgumentParser(
        description="CAIDBenchmark online continual fake detection",
        parents=[pre_parser],
    )
    parser.set_defaults(method=method)

    # ========== Experiment configuration ==========
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.get("seeds", [1]))
    parser.add_argument("--note", type=str, default=defaults.get("note", ""), help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default=defaults.get("log_path", "run_logs"), help="Root directory for run logs and outputs.")
    parser.add_argument("--use_swanlab", "--swanlab", dest="use_swanlab",
                        action="store_true", default=defaults.get("use_swanlab", True),
                        help="Enable SwanLab experiment tracking. Enabled by default.")
    parser.add_argument("--no_swanlab", dest="use_swanlab",
                        action="store_false",
                        help="Disable SwanLab experiment tracking.")
    parser.add_argument("--swanlab_project", type=str, default="ocl4aid",
                        help="SwanLab project name.")
    parser.add_argument("--swanlab_workspace", type=str, default=None,
                        help="SwanLab workspace/organization username. Defaults to personal workspace.")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None,
                        help="SwanLab experiment name. Defaults to '<note_or_method>_<YYYYmmdd_HHMMSS>'.")
    parser.add_argument("--swanlab_description", type=str, default=None,
                        help="SwanLab experiment description.")
    parser.add_argument("--swanlab_group", type=str, default=None,
                        help="SwanLab experiment group.")
    parser.add_argument("--swanlab_tags", nargs="*", default=None,
                        help="SwanLab experiment tags.")
    parser.add_argument("--swanlab_mode", type=str, default="cloud",
                        choices=["cloud", "local", "offline", "disabled"],
                        help="SwanLab logging mode.")
    parser.add_argument("--swanlab_logdir", type=str, default=None,
                        help="Directory for SwanLab local/offline logs. Defaults to the run log directory.")
    parser.add_argument("--swanlab_public", action="store_true", default=False,
                        help="Create the SwanLab project as public when applicable.")

    # ============ Model configuration =============
    parser.add_argument("--backbone", type=str, default=defaults.get("backbone", "vit_base_patch16_224"), help="Backbone name")
    parser.add_argument("--pretrained", action="store_true", default=defaults.get("pretrained", True),
                        help="Load pretrained backbone weights. Enabled by default.")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                        help="Disable pretrained backbone weights, useful for local smoke tests.")

    # =========== Dataset configuration ============
    parser.add_argument("--caidbench_data_dir", type=str, required=defaults.get("caidbench_data_dir") is None,
                        default=defaults.get("caidbench_data_dir"),
                        help="Root directory of the CAIDBenchmark Arrow package.")
    parser.add_argument("--caidbench_protocol", type=str,
                        default=defaults.get("caidbench_protocol", "protocol_presets/caidbench/model_appearance_order_protocol.yaml"),
                        help="CAIDBenchmark continual protocol YAML.")
    parser.add_argument("--caidbench_index_path", type=str, default=defaults.get("caidbench_index_path"),
                        help="Optional CAIDBenchmark index parquet override. Defaults to protocol index_path.")
    parser.add_argument("--caidbench_image_column", type=str, default=defaults.get("caidbench_image_column", "image"),
                        help="Image column name in CAIDBenchmark Arrow files.")
    parser.add_argument("--caidbench_face_bbox_path", type=str, default=defaults.get("caidbench_face_bbox_path"),
                        help="Optional face bbox parquet. Defaults to forgerynet_face_bboxes_all_generators.parquet in the CAIDBenchmark root when present.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default=defaults.get("opt_name", "sgd"), help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default=defaults.get("sched_name", "default"), help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=defaults.get("use_amp", False), help="Use automatic mixed precision.")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=defaults.get("n_worker", 0), help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=defaults.get("batchsize", 16),
                        help="Global online batch size. In DDP it is split evenly across ranks.")
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 0.05), help="learning rate")
    parser.add_argument("--online_iter", type=float, default=defaults.get("online_iter", 1), help="number of model updates per samples seen.")
    parser.add_argument("--base_stage_epochs", type=int, default=defaults.get("base_stage_epochs", 1),
                        help="Supervised epochs on protocol stage 0 before online continual learning starts. Set 0 to disable.")
    parser.add_argument("--save_base_checkpoint", action="store_true",
                        default=defaults.get("save_base_checkpoint", False),
                        help="Save a reusable checkpoint after the supervised base stage.")
    parser.add_argument("--no_save_base_checkpoint", dest="save_base_checkpoint",
                        action="store_false",
                        help="Disable base-stage checkpoint saving even if enabled by YAML.")
    parser.add_argument("--base_checkpoint_dir", type=str,
                        default=defaults.get("base_checkpoint_dir"),
                        help="Directory for automatic base-stage checkpoints. Defaults to <log_path>/base_checkpoints.")
    parser.add_argument("--load_base_checkpoint", type=str,
                        default=defaults.get("load_base_checkpoint"),
                        help="Load a saved base-stage checkpoint before online learning. Use 'auto' for the default path.")
    parser.add_argument("--base_checkpoint_only", action="store_true",
                        default=defaults.get("base_checkpoint_only", False),
                        help="Stop after loading or saving the base stage; useful for precomputing reusable bases.")
    parser.add_argument("--no_base_checkpoint_only", dest="base_checkpoint_only",
                        action="store_false",
                        help="Disable base-stage-only mode even if enabled by YAML.")
    parser.add_argument("--stage_blurry_n", "--blurry_n", "--n", dest="stage_blurry_n",
                        type=int, default=defaults.get("stage_blurry_n", 100),
                        help="Percent of each protocol stage kept as hard-boundary home samples. 100 recovers the strict stream.")
    parser.add_argument("--stage_blurry_m", "--blurry_m", "--m", dest="stage_blurry_m",
                        type=int, default=defaults.get("stage_blurry_m", 0),
                        help="Percent of non-home eligible samples leaked to adjacent time stages. 0 recovers the strict stream.")

    parser.add_argument("--transforms", nargs="*", default=defaults.get("transforms", ["autoaug"]), help="Additional train transforms [cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=defaults.get("no_batchmask", False), help="Disable batch mask, use seen mask")

    parser.add_argument("--topk", type=int, default=defaults.get("topk", 1), help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_interval", type=int, default=defaults.get("eval_interval", 20000),
                        help="Online-phase stream evaluation interval in training samples. <=0 disables periodic stream eval.")

    args, extra_cli_args = parser.parse_known_args()
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    for key, value in _parse_extra_cli_overrides(extra_cli_args, defaults).items():
        setattr(args, key, value)
    if args.method is None:
        args.method = method
    return args
