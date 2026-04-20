from .OpenFakeProtocol import OpenFakeProtocol
from .OnlineIterDataset import OnlineIterDataset

DATASETS = {
    "openfake_protocol": OpenFakeProtocol,
}

__all__ = [cls.__name__ for cls in DATASETS.values()] + ["DATASETS", "OnlineIterDataset"]
