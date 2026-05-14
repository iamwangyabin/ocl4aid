import argparse

from datasets import DATASETS
from methods import METHODS


def base_parser():
    parser = argparse.ArgumentParser(description="OpenFake protocol continual fake detection")

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
    parser.add_argument("--dataset", type=str, default="openfake_protocol", help="dataset name", choices=DATASETS.keys())
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset root directory for protocol image paths. Auto-set for Hugging Face OpenFake when omitted.")
    parser.add_argument("--protocol_manifest", type=str, default=None,
                        help="Path to stage_manifest.json for openfake_protocol. If omitted, OpenFake is prepared from Hugging Face.")
    parser.add_argument("--auto_openfake_hf", action="store_true", default=True,
                        help="Automatically prepare OpenFake from Hugging Face when protocol_manifest is omitted.")
    parser.add_argument("--no_auto_openfake_hf", dest="auto_openfake_hf",
                        action="store_false",
                        help="Disable automatic OpenFake Hugging Face preparation.")
    parser.add_argument("--openfake_hf_dataset_id", type=str, default="ComplexDataLab/OpenFake",
                        help="Hugging Face dataset id for automatic OpenFake preparation.")
    parser.add_argument("--openfake_hf_config", type=str, default="core",
                        help="Hugging Face dataset config for automatic OpenFake preparation.")
    parser.add_argument("--openfake_hf_split", type=str, default="train",
                        help="Hugging Face split for automatic OpenFake preparation.")
    parser.add_argument("--openfake_cache_dir", type=str, default=None,
                        help="Optional cache dir for auto-prepared OpenFake protocol files. Defaults to Hugging Face datasets cache.")
    parser.add_argument("--openfake_hf_cache_dir", type=str, default=None,
                        help="Optional cache_dir passed to Hugging Face load_dataset for non-default dataset cache locations.")
    parser.add_argument("--openfake_generators", nargs="*", default=None,
                        help="OpenFake generator names to include when auto-preparing from Hugging Face.")
    parser.add_argument("--openfake_fake_train_per_generator", type=int, default=8,
                        help="Fake train samples per generator for automatic OpenFake preparation.")
    parser.add_argument("--openfake_fake_test_per_generator", type=int, default=2,
                        help="Fake test samples per generator for automatic OpenFake preparation.")
    parser.add_argument("--openfake_real_train", type=int, default=32,
                        help="Real train samples for automatic OpenFake preparation.")
    parser.add_argument("--openfake_real_test", type=int, default=8,
                        help="Real test samples for automatic OpenFake preparation.")
    parser.add_argument("--openfake_hf_streaming", action="store_true", default=False,
                        help="Stream OpenFake from Hugging Face instead of using the default download/cache path.")
    parser.add_argument("--no_openfake_hf_streaming", dest="openfake_hf_streaming",
                        action="store_false",
                        help="Use Hugging Face's default non-streaming download/cache path.")
    parser.add_argument("--openfake_force_prepare", action="store_true", default=False,
                        help="Rebuild the auto-prepared OpenFake cache even if manifest files already exist.")
    parser.add_argument("--openfake_protocol_seed", type=int, default=13,
                        help="Protocol split seed for automatic OpenFake preparation.")
    parser.add_argument("--n_tasks", type=int, default=29, help="The number of stages; overridden by the protocol manifest.")
    parser.add_argument("--step_num", type=int, default=-1,
                        help="Number of internal steps for task-free prompt methods; if <=0, defaults to n_tasks.")

    parser.add_argument("--n", type=int, default=50, help="Unused for openfake_protocol; kept for method compatibility.")
    parser.add_argument("--m", type=int, default=10, help="Unused for openfake_protocol; kept for method compatibility.")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="Unused for openfake_protocol; kept for method compatibility.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch.")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'autoaug'], help="Additional train transforms [cutmix, cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=False, help="Disable batch mask, use seen mask")

    # ========== Evaluation configuration ==========
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

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

    # ======== Expert similarity analysis ==========
    parser.add_argument("--analysis_expert_similarity", action="store_true", default=False,
                        help="If set, run expert feature similarity / CKA (including residual vs common) analysis after training.")

    args = parser.parse_args()
    return args
