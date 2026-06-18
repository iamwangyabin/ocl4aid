from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from torch.utils.data._utils.collate import default_collate


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BadSample:
    index: int
    error: str


def make_bad_sample(index: int, exc: Exception) -> BadSample:
    message = f"{type(exc).__name__}: {exc}"
    logger.warning("Skipping unreadable CAIDBenchmark sample index=%s: %s", index, message)
    return BadSample(index=int(index), error=message)


def safe_collate_drop_bad(batch: list[Any]):
    valid = [item for item in batch if not isinstance(item, BadSample)]
    if len(valid) != len(batch):
        bad_items = [item for item in batch if isinstance(item, BadSample)]
        preview = ", ".join(f"{item.index} ({item.error})" for item in bad_items[:3])
        if len(bad_items) > 3:
            preview += f", ... +{len(bad_items) - 3} more"
        logger.warning("Dropped %s unreadable samples from batch: %s", len(bad_items), preview)
    if not valid:
        return None
    return default_collate(valid)
