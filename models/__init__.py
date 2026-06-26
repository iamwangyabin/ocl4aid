from .codaprompt import CodaPrompt
from .dualprompt import DualPrompt
from .flyprompt import FlyPrompt
from .l2p import L2P
from .mvp import MVP
from .ranpac import RanPAC
from .hide_norga_prefix_vit import HiDePrefixModel, NoRGaPrefixModel
from .hide_lora_vit import HiDeLoRAModel
from .hide_adapter_vit import HiDeAdapterModel
from .online_lora import OnlineLoRAModel
from .sdlora import SDLoRAModel
from .singleprompt import SinglePrompt
from .sprompt import SPrompt
from .rineside_gauss import RineSideGauss
from .rigev1 import RIGEv1
from .rigev2 import RIGEv2

MODELS = {
    "codaprompt": CodaPrompt,
    "dualprompt": DualPrompt,
    "flyprompt": FlyPrompt,
    "l2p": L2P,
    "mvp": MVP,
    "ranpac": RanPAC,
    "hide": HiDePrefixModel,
    "hide_lora": HiDeLoRAModel,
    "hide_adapter": HiDeAdapterModel,
    "norga": NoRGaPrefixModel,
    "online_lora": OnlineLoRAModel,
    "sdlora": SDLoRAModel,
    "singleprompt": SinglePrompt,
    "sprompt": SPrompt,
    "rineside_gauss": RineSideGauss,
    "rigev1": RIGEv1,
    "rigev2": RIGEv2,
}
