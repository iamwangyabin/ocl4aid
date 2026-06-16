import argparse

from methods import METHODS


def base_parser():
    parser = argparse.ArgumentParser(description="CAIDBenchmark online continual fake detection")

    # ========== Experiment configuration ==========
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--note", type=str, default="", help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")
    parser.add_argument("--use_swanlab", "--swanlab", dest="use_swanlab",
                        action="store_true", default=True,
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
    parser.add_argument("--method", type=str, default="l2p", help="Method name", choices=METHODS.keys())
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone name")

    # =========== Dataset configuration ============
    parser.add_argument("--caidbench_data_dir", type=str, required=True,
                        help="Root directory of the CAIDBenchmark Arrow package.")
    parser.add_argument("--caidbench_protocol", type=str,
                        default="protocol_presets/caidbench/model_appearance_order_protocol.yaml",
                        help="CAIDBenchmark continual protocol YAML.")
    parser.add_argument("--caidbench_index_path", type=str, default=None,
                        help="Optional CAIDBenchmark index parquet override. Defaults to protocol index_path.")
    parser.add_argument("--caidbench_label_mode", type=str, default="generator",
                        choices=["generator", "binary"],
                        help="Use real+generator classes or binary real/fake labels for training.")
    parser.add_argument("--caidbench_image_column", type=str, default="image",
                        help="Image column name in CAIDBenchmark Arrow files.")
    parser.add_argument("--step_num", type=int, default=-1,
                        help="Number of internal steps for task-free prompt methods; if <=0, defaults to n_tasks.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--base_epochs", type=int, default=1,
                        help="Number of epochs for the base session only. Online stages are single-pass.")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=["autoaug"], help="Additional train transforms [cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=False, help="Disable batch mask, use seen mask")

    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")

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
    parser.add_argument("--lam_orth", type=float, default=1, help="Orthogonal loss weight for HiDe/NoRGa.")
    parser.add_argument("--ca_num_per_class", type=int, default=200, help="Number of CA samples per class for HiDe/NoRGa.")
    parser.add_argument("--ca_steps", type=int, default=200, help="Number of CA optimization steps for HiDe/NoRGa.")

    # ========== SD-LoRA configurations ==========
    parser.add_argument("--sdlora_rank", type=int, default=10, help="LoRA rank for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_alpha", type=float, default=0.8, help="Scaling factor alpha for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_layers", type=str, default="all", help="Which ViT blocks to apply LoRA to (e.g., 'all', 'last4').")
    parser.add_argument("--sdlora_ortho_weight", type=float, default=0.0, help="Orthogonal loss weight for SD-LoRA (0 means disabled).")

    # ========== FlyPrompt configurations ==========
    parser.add_argument("--len_prompt", type=int, default=20, help="The length of the prompt for each expert")
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="The position of the prompt")
    parser.add_argument("--logit_type", type=str, default="cos_sim", choices=["linear", "cos_sim"],
                        help="Classifier logit type for SinglePrompt.")
    parser.add_argument("--rp_dim", type=int, default=10000, help="The dimension of the random projection head")
    parser.add_argument("--rp_ridge", type=float, default=1e4, help="The ridge parameter for the random projection head")
    parser.add_argument("--ema_ratio", type=float, nargs="+", default=[0.9, 0.99], help="The EMA ratio for the expert FCs")
    parser.add_argument("--ensemble_method", type=str, default="softmax_max_prob", choices=["mean", "max_prob", "min_entropy", "softmax_mean", "softmax_max_prob", "softmax_min_entropy"],
                        help="Ensemble method for combining expert outputs: mean (average), max (maximum), min_entropy (minimum entropy), and softmax variants of these.")

    # ========== RPFC gating configurations ==========
    parser.add_argument("--use_rp_gate", action="store_true", default=False,
                        help="Use FlyPrompt-style RPFC head for task gating in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ========== EMA head bank configurations ==========
    parser.add_argument("--use_ema_head", action="store_true", default=False,
                        help="Use EMA-based classifier head bank and ensemble in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    args = parser.parse_args()
    return args
