from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .ranpac import RanPAC
from .slca import SLCA
from .hide_norga_trainer import HiDeGCLTrainer, NoRGaGCLTrainer
from .online_lora import OnlineLoRA
from .sdlora import SDLoRAGCL
from .singleprompt import SinglePromptTrainer
from .sprompt import SPrompt as SPromptTrainer
from .rigev1 import RIGEv1
from .rigev2 import RIGEv2

METHODS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "ranpac": RanPAC,
    "slca": SLCA,
    "hide": HiDeGCLTrainer,
    "hide_lora": HiDeGCLTrainer,
    "hide_adapter": HiDeGCLTrainer,
    "norga": NoRGaGCLTrainer,
    "online_lora": OnlineLoRA,
    "sdlora": SDLoRAGCL,
    "singleprompt": SinglePromptTrainer,
    "sprompt": SPromptTrainer,
    "rigev1": RIGEv1,
    "rigev2": RIGEv2,
}
