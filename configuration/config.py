import argparse
from pathlib import Path

from methods import METHODS


DEFAULT_FRAMEWORK_CONFIG = "configs/framework/caidbench.yaml"


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
        ("run", "seeds"): "seeds",
        ("run", "note"): "note",
        ("run", "log_dir"): "log_path",
        ("model", "backbone"): "backbone",
        ("train", "optimizer"): "opt_name",
        ("train", "scheduler"): "sched_name",
        ("train", "amp"): "use_amp",
        ("train", "workers"): "n_worker",
        ("train", "batch_size"): "batchsize",
        ("train", "lr"): "lr",
        ("train", "online_iter"): "online_iter",
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
    payload = _load_yaml(Path("configs/methods") / f"{method}.yaml")
    return payload


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

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default=defaults.get("opt_name", "sgd"), help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default=defaults.get("sched_name", "default"), help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=defaults.get("use_amp", False), help="Use automatic mixed precision.")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=defaults.get("n_worker", 0), help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=defaults.get("batchsize", 16), help="batch size")
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 0.05), help="learning rate")
    parser.add_argument("--online_iter", type=float, default=defaults.get("online_iter", 1), help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=defaults.get("transforms", ["autoaug"]), help="Additional train transforms [cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=defaults.get("no_batchmask", False), help="Disable batch mask, use seen mask")

    parser.add_argument("--topk", type=int, default=defaults.get("topk", 1), help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_interval", type=int, default=defaults.get("eval_interval", 20000),
                        help="Online-phase stream evaluation interval in training samples. <=0 disables periodic stream eval.")

    # ============= ViT configurations =============
    parser.add_argument('--profile', action='store_true', default=False, help='enable profiling for ViT_Prompt')

    # ============= MISA configurations ============
    parser.add_argument('--load_pt', action='store_true', default=False, help='load pretrained prompts (MISA)')

    # ============= MePo configurations ============
    parser.add_argument('--mepo_backbone_path', type=str, default=None,
                        help='Path to pretrained backbone checkpoint for MEPO backbone override.')
    parser.add_argument('--cov_path', type=str, default=None,
                        help='Path to covariance matrix .npy for MEPO CLS calibration.')
    parser.add_argument('--cov_coef', type=float, default=0.7,
                        help='Interpolation coeff between original and MEPO-calibrated CLS (0-1).')

    # ======== HiDe / NoRGa configurations =========
    parser.add_argument("--lam_orth", type=float, default=defaults.get("lam_orth", 1), help="Orthogonal loss weight for HiDe/NoRGa.")
    parser.add_argument("--ca_num_per_class", type=int, default=defaults.get("ca_num_per_class", 200), help="Number of CA samples per class for HiDe/NoRGa.")
    parser.add_argument("--ca_steps", type=int, default=defaults.get("ca_steps", 200), help="Number of CA optimization steps for HiDe/NoRGa.")

    # ========== SD-LoRA configurations ==========
    parser.add_argument("--sdlora_rank", type=int, default=defaults.get("sdlora_rank", 10), help="LoRA rank for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_alpha", type=float, default=defaults.get("sdlora_alpha", 0.8), help="Scaling factor alpha for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_layers", type=str, default=defaults.get("sdlora_layers", "all"), help="Which ViT blocks to apply LoRA to (e.g., 'all', 'last4').")
    parser.add_argument("--sdlora_ortho_weight", type=float, default=defaults.get("sdlora_ortho_weight", 0.0), help="Orthogonal loss weight for SD-LoRA (0 means disabled).")

    # ========== FlyPrompt configurations ==========
    parser.add_argument("--len_prompt", type=int, default=defaults.get("len_prompt", 20), help="The length of the prompt for each expert")
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=defaults.get("pos_prompt", [0, 1, 2, 3, 4]), help="The position of the prompt")
    parser.add_argument("--logit_type", type=str, default=defaults.get("logit_type", "cos_sim"), choices=["linear", "cos_sim"],
                        help="Classifier logit type for SinglePrompt.")
    parser.add_argument("--rp_dim", type=int, default=defaults.get("rp_dim", 10000), help="The dimension of the random projection head")
    parser.add_argument("--rp_ridge", type=float, default=defaults.get("rp_ridge", 1e4), help="The ridge parameter for the random projection head")
    parser.add_argument("--ema_ratio", type=float, nargs="+", default=defaults.get("ema_ratio", [0.9, 0.99]), help="The EMA ratio for the expert FCs")
    parser.add_argument("--ensemble_method", type=str, default=defaults.get("ensemble_method", "softmax_max_prob"), choices=["mean", "max_prob", "min_entropy", "softmax_mean", "softmax_max_prob", "softmax_min_entropy"],
                        help="Ensemble method for combining expert outputs: mean (average), max (maximum), min_entropy (minimum entropy), and softmax variants of these.")

    # ========== RPFC gating configurations ==========
    parser.add_argument("--use_rp_gate", action="store_true", default=defaults.get("use_rp_gate", False),
                        help="Use FlyPrompt-style RPFC head for task gating in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ========== EMA head bank configurations ==========
    parser.add_argument("--use_ema_head", action="store_true", default=defaults.get("use_ema_head", False),
                        help="Use EMA-based classifier head bank and ensemble in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    args = parser.parse_args()
    if args.method is None:
        args.method = method
    return args
