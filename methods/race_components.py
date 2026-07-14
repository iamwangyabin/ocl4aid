"""Boundary-free RACE detector, replay, and expert-allocation components.

The components in this module deliberately do not depend on protocol stages,
datasets, trainers, or CUDA.  They operate only on learner-visible compressed
features, frozen base logits, binary labels, and causal sample offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Sequence

import torch


def _cpu_float_matrix(value, *, columns: Optional[int] = None, name: str = "value"):
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor, got shape {tuple(tensor.shape)}")
    if columns is not None and int(tensor.size(1)) != int(columns):
        raise ValueError(
            f"{name} must have {columns} columns, got {int(tensor.size(1))}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


def _cpu_binary_labels(value, *, rows: Optional[int] = None):
    labels = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if rows is not None and int(labels.numel()) != int(rows):
        raise ValueError(f"labels must have {rows} rows, got {int(labels.numel())}")
    if labels.numel() and not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("RACE components require binary labels in {0, 1}")
    return labels.contiguous()


@dataclass
class DiagonalMoments:
    """Streaming diagonal moments represented with Welford's ``count/mean/m2``."""

    count: int = 0
    mean: Optional[torch.Tensor] = None
    m2: Optional[torch.Tensor] = None

    def update(self, features) -> "DiagonalMoments":
        features = _cpu_float_matrix(features, name="features")
        if features.size(0) == 0:
            return self
        batch_count = int(features.size(0))
        batch_mean = features.mean(dim=0)
        centered = features - batch_mean
        batch_m2 = centered.square().sum(dim=0)
        self.merge(DiagonalMoments(batch_count, batch_mean, batch_m2))
        return self

    def merge(self, other: "DiagonalMoments") -> "DiagonalMoments":
        if int(other.count) <= 0:
            return self
        if other.mean is None or other.m2 is None:
            raise ValueError("non-empty moments require mean and m2")
        other_mean = torch.as_tensor(other.mean).detach().cpu().float().reshape(-1)
        other_m2 = torch.as_tensor(other.m2).detach().cpu().float().reshape(-1)
        if other_mean.shape != other_m2.shape:
            raise ValueError("mean and m2 shapes must match")
        if self.count <= 0:
            self.count = int(other.count)
            self.mean = other_mean.clone()
            self.m2 = other_m2.clone()
            return self
        if self.mean is None or self.m2 is None:
            raise ValueError("non-empty moments require mean and m2")
        if self.mean.shape != other_mean.shape:
            raise ValueError("cannot merge moments with different feature dimensions")
        count = int(self.count)
        other_count = int(other.count)
        total = count + other_count
        delta = other_mean - self.mean
        self.mean = self.mean + delta * (other_count / total)
        self.m2 = (
            self.m2
            + other_m2
            + delta.square() * (count * other_count / total)
        )
        self.count = total
        return self

    @classmethod
    def from_features(cls, features) -> "DiagonalMoments":
        return cls().update(features)

    def variance(self, floor: float = 0.0, *, unbiased: bool = True) -> torch.Tensor:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("variance is undefined for empty moments")
        denominator = max(self.count - 1, 1) if unbiased else max(self.count, 1)
        return (self.m2 / denominator).clamp_min(float(floor))

    def clone(self) -> "DiagonalMoments":
        return DiagonalMoments(
            int(self.count),
            None if self.mean is None else self.mean.clone(),
            None if self.m2 is None else self.m2.clone(),
        )

    def state_dict(self) -> dict:
        return {
            "count": int(self.count),
            "mean": None if self.mean is None else self.mean.clone(),
            "m2": None if self.m2 is None else self.m2.clone(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DiagonalMoments":
        count = int(state.get("count", 0))
        mean = state.get("mean")
        m2 = state.get("m2")
        if count <= 0:
            return cls()
        if not torch.is_tensor(mean) or not torch.is_tensor(m2):
            raise ValueError("serialized non-empty moments require tensor mean and m2")
        return cls(count, mean.detach().cpu().float().clone(), m2.detach().cpu().float().clone())


def symmetric_diagonal_gaussian_kl(
    left: DiagonalMoments,
    right: DiagonalMoments,
    *,
    variance_floor: float = 1e-4,
) -> float:
    """Return symmetric diagonal-Gaussian KL averaged over feature dimensions."""

    if left.mean is None or right.mean is None:
        raise ValueError("Gaussian KL requires non-empty moments")
    if left.mean.shape != right.mean.shape:
        raise ValueError("Gaussian KL requires matching feature dimensions")
    left_var = left.variance(variance_floor)
    right_var = right.variance(variance_floor)
    delta2 = (left.mean - right.mean).square()
    # 0.5 * (KL(P||Q) + KL(Q||P)), divided by dimensionality.
    value = 0.25 * torch.mean(
        left_var / right_var
        + right_var / left_var
        + delta2 * (left_var.reciprocal() + right_var.reciprocal())
        - 2.0
    )
    return float(torch.nan_to_num(value, nan=math.inf, posinf=math.inf).item())


@dataclass(frozen=True)
class ChangeEvent:
    sample_offset: int
    raw_score: float
    ewma_score: float
    threshold: float
    persistence: int
    valid_classes: tuple[int, ...]
    candidate_features: torch.Tensor
    candidate_labels: torch.Tensor


class ClassConditionalChangeDetector:
    """Causal class-conditional two-window Gaussian change detector."""

    STATE_VERSION = 1

    def __init__(
        self,
        feature_dim: int,
        *,
        window_size: int = 512,
        ewma_beta: float = 0.9,
        false_alarm_rate: float = 0.05,
        persistence: int = 3,
        warmup_samples: Optional[int] = None,
        cooldown_samples: Optional[int] = None,
        min_class_count: int = 16,
        min_valid_classes: int = 2,
        class_conditional: bool = True,
        variance_floor: float = 1e-4,
        threshold: Optional[float] = None,
        min_calibration_scores: int = 20,
        normalize_features: bool = True,
    ):
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.ewma_beta = float(ewma_beta)
        self.false_alarm_rate = float(false_alarm_rate)
        self.persistence_required = int(persistence)
        self.warmup_samples = int(
            self.window_size if warmup_samples is None else warmup_samples
        )
        self.cooldown_samples = int(
            2 * self.window_size if cooldown_samples is None else cooldown_samples
        )
        self.min_class_count = int(min_class_count)
        self.min_valid_classes = int(min_valid_classes)
        self.class_conditional = bool(class_conditional)
        self.variance_floor = float(variance_floor)
        self.threshold = None if threshold is None else float(threshold)
        self.min_calibration_scores = int(min_calibration_scores)
        self.normalize_features = bool(normalize_features)
        self._validate_configuration()

        self._reference: Dict[int, DiagonalMoments] = {}
        self._recent_features = torch.empty((0, self.feature_dim), dtype=torch.float32)
        self._recent_labels = torch.empty((0,), dtype=torch.long)
        self._null_scores: list[float] = []
        self._ewma_score = 0.0
        self._persistence_count = 0
        self._samples_since_reference = 0
        self._cooldown_remaining = 0
        self._awaiting_reset = False
        self._last_raw_score: Optional[float] = None
        self._last_valid_classes: tuple[int, ...] = ()
        self._trigger_count = 0

    def _validate_configuration(self):
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= self.ewma_beta < 1.0:
            raise ValueError("ewma_beta must be in [0, 1)")
        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if self.persistence_required <= 0:
            raise ValueError("persistence must be positive")
        if self.warmup_samples < 0 or self.cooldown_samples < 0:
            raise ValueError("warmup_samples and cooldown_samples must be non-negative")
        if self.min_class_count <= 0:
            raise ValueError("min_class_count must be positive")
        maximum_classes = 2 if self.class_conditional else 1
        if not 1 <= self.min_valid_classes <= maximum_classes:
            raise ValueError(
                f"min_valid_classes must be in 1..{maximum_classes} for this detector"
            )
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        if self.min_calibration_scores <= 0:
            raise ValueError("min_calibration_scores must be positive")

    def _prepare_features(self, features) -> torch.Tensor:
        features = _cpu_float_matrix(
            features,
            columns=self.feature_dim,
            name="features",
        )
        if self.normalize_features and features.numel():
            features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return features

    def _group_moments(self, features, labels) -> Dict[int, DiagonalMoments]:
        features = self._prepare_features(features)
        labels = _cpu_binary_labels(labels, rows=features.size(0))
        if not self.class_conditional:
            return {0: DiagonalMoments.from_features(features)} if features.numel() else {}
        result = {}
        for label in (0, 1):
            selected = features[labels == label]
            if selected.size(0):
                result[label] = DiagonalMoments.from_features(selected)
        return result

    def _score_moments(
        self,
        left: Dict[int, DiagonalMoments],
        right: Dict[int, DiagonalMoments],
    ) -> tuple[Optional[float], tuple[int, ...]]:
        valid = tuple(
            label
            for label in sorted(set(left).intersection(right))
            if left[label].count >= self.min_class_count
            and right[label].count >= self.min_class_count
        )
        if len(valid) < self.min_valid_classes:
            return None, valid
        scores = [
            symmetric_diagonal_gaussian_kl(
                left[label],
                right[label],
                variance_floor=self.variance_floor,
            )
            for label in valid
        ]
        score = float(sum(scores) / len(scores))
        return score, valid

    def calibrate(self, features, labels) -> float:
        """Set the stable base reference and calibrate tau from adjacent windows."""

        features = self._prepare_features(features)
        labels = _cpu_binary_labels(labels, rows=features.size(0))
        self._reference = self._group_moments(features, labels)

        blocks = []
        for start in range(0, int(features.size(0)) - self.window_size + 1, self.window_size):
            stop = start + self.window_size
            blocks.append(self._group_moments(features[start:stop], labels[start:stop]))
        null_scores = []
        for left, right in zip(blocks[:-1], blocks[1:]):
            score, _valid = self._score_moments(left, right)
            if score is not None and math.isfinite(score):
                null_scores.append(float(score))
        if len(null_scores) < self.min_calibration_scores:
            raise ValueError(
                "insufficient valid base-window discrepancies for calibration: "
                f"need {self.min_calibration_scores}, got {len(null_scores)}"
            )
        quantile = 1.0 - self.false_alarm_rate
        values = torch.tensor(null_scores, dtype=torch.float64)
        try:
            threshold = torch.quantile(values, quantile, interpolation="higher")
        except TypeError:
            threshold = torch.quantile(values, quantile)
        self.threshold = float(threshold.item())
        self._null_scores = null_scores
        self._reset_monitoring(clear_recent=True, cooldown=0)
        return self.threshold

    def reset_reference(self, features, labels, *, start_cooldown: bool = True):
        """Start a new internal regime from a causal candidate window."""

        features = self._prepare_features(features)
        labels = _cpu_binary_labels(labels, rows=features.size(0))
        reference = self._group_moments(features, labels)
        valid_reference_classes = sum(
            moments.count >= self.min_class_count for moments in reference.values()
        )
        if valid_reference_classes < self.min_valid_classes:
            raise ValueError("candidate window does not contain enough class support")
        self._reference = reference
        self._reset_monitoring(
            clear_recent=True,
            cooldown=self.cooldown_samples if start_cooldown else 0,
        )

    def _reset_monitoring(self, *, clear_recent: bool, cooldown: int):
        if clear_recent:
            self._recent_features = torch.empty(
                (0, self.feature_dim), dtype=torch.float32
            )
            self._recent_labels = torch.empty((0,), dtype=torch.long)
        self._ewma_score = 0.0
        self._persistence_count = 0
        self._samples_since_reference = 0
        self._cooldown_remaining = int(cooldown)
        self._awaiting_reset = False
        self._last_raw_score = None
        self._last_valid_classes = ()

    def _append_recent(self, features, labels):
        self._recent_features = torch.cat([self._recent_features, features], dim=0)
        self._recent_labels = torch.cat([self._recent_labels, labels], dim=0)
        if self._recent_labels.numel() > self.window_size:
            self._recent_features = self._recent_features[-self.window_size :]
            self._recent_labels = self._recent_labels[-self.window_size :]

    def observe(self, features, labels, sample_offset: int) -> Optional[ChangeEvent]:
        """Observe one causal batch and return a change event only on a trigger."""

        if not self._reference:
            raise RuntimeError("detector must be calibrated or given a reference first")
        features = self._prepare_features(features)
        labels = _cpu_binary_labels(labels, rows=features.size(0))
        if features.size(0) == 0:
            return None
        self._append_recent(features, labels)
        batch_size = int(features.size(0))
        self._samples_since_reference += batch_size
        self._cooldown_remaining = max(0, self._cooldown_remaining - batch_size)

        recent = self._group_moments(self._recent_features, self._recent_labels)
        score, valid_classes = self._score_moments(self._reference, recent)
        self._last_valid_classes = valid_classes
        if score is None:
            self._last_raw_score = None
            self._persistence_count = 0
            return None

        self._last_raw_score = float(score)
        self._ewma_score = (
            self.ewma_beta * self._ewma_score
            + (1.0 - self.ewma_beta) * float(score)
        )
        eligible = (
            not self._awaiting_reset
            and self.threshold is not None
            and self._samples_since_reference >= self.warmup_samples
            and self._cooldown_remaining == 0
        )
        if eligible and self._ewma_score > float(self.threshold):
            self._persistence_count += 1
        else:
            self._persistence_count = 0
        if self._persistence_count < self.persistence_required:
            return None

        self._awaiting_reset = True
        self._trigger_count += 1
        return ChangeEvent(
            sample_offset=int(sample_offset),
            raw_score=float(score),
            ewma_score=float(self._ewma_score),
            threshold=float(self.threshold),
            persistence=int(self._persistence_count),
            valid_classes=valid_classes,
            candidate_features=self._recent_features.clone(),
            candidate_labels=self._recent_labels.clone(),
        )

    def recent_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._recent_features.clone(), self._recent_labels.clone()

    def diagnostics(self) -> dict:
        label_counts = {
            str(label): int((self._recent_labels == label).sum().item())
            for label in (0, 1)
        }
        return {
            "threshold": self.threshold,
            "null_score_count": len(self._null_scores),
            "last_raw_score": self._last_raw_score,
            "ewma_score": float(self._ewma_score),
            "persistence_count": int(self._persistence_count),
            "samples_since_reference": int(self._samples_since_reference),
            "cooldown_remaining": int(self._cooldown_remaining),
            "recent_count": int(self._recent_labels.numel()),
            "recent_label_counts": label_counts,
            "valid_classes": list(self._last_valid_classes),
            "trigger_count": int(self._trigger_count),
            "awaiting_reference_reset": bool(self._awaiting_reset),
        }

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "configuration": {
                "feature_dim": self.feature_dim,
                "window_size": self.window_size,
                "ewma_beta": self.ewma_beta,
                "false_alarm_rate": self.false_alarm_rate,
                "persistence": self.persistence_required,
                "warmup_samples": self.warmup_samples,
                "cooldown_samples": self.cooldown_samples,
                "min_class_count": self.min_class_count,
                "min_valid_classes": self.min_valid_classes,
                "class_conditional": self.class_conditional,
                "variance_floor": self.variance_floor,
                "min_calibration_scores": self.min_calibration_scores,
                "normalize_features": self.normalize_features,
            },
            "threshold": self.threshold,
            "null_scores": list(self._null_scores),
            "reference": {
                int(label): moments.state_dict()
                for label, moments in self._reference.items()
            },
            "recent_features": self._recent_features.clone(),
            "recent_labels": self._recent_labels.clone(),
            "ewma_score": float(self._ewma_score),
            "persistence_count": int(self._persistence_count),
            "samples_since_reference": int(self._samples_since_reference),
            "cooldown_remaining": int(self._cooldown_remaining),
            "awaiting_reset": bool(self._awaiting_reset),
            "last_raw_score": self._last_raw_score,
            "last_valid_classes": list(self._last_valid_classes),
            "trigger_count": int(self._trigger_count),
        }

    def load_state_dict(self, state: dict):
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported detector state version")
        saved_configuration = state.get("configuration", {})
        current_configuration = self.state_dict()["configuration"]
        if saved_configuration != current_configuration:
            raise ValueError("detector state configuration does not match current configuration")
        self.threshold = (
            None if state.get("threshold") is None else float(state["threshold"])
        )
        self._null_scores = [float(value) for value in state.get("null_scores", [])]
        self._reference = {
            int(label): DiagonalMoments.from_state_dict(payload)
            for label, payload in state.get("reference", {}).items()
        }
        self._recent_features = _cpu_float_matrix(
            state.get("recent_features", torch.empty((0, self.feature_dim))),
            columns=self.feature_dim,
            name="recent_features",
        )
        self._recent_labels = _cpu_binary_labels(
            state.get("recent_labels", torch.empty(0)),
            rows=self._recent_features.size(0),
        )
        if self._recent_labels.numel() > self.window_size:
            raise ValueError("serialized recent window exceeds configured capacity")
        self._ewma_score = float(state.get("ewma_score", 0.0))
        self._persistence_count = int(state.get("persistence_count", 0))
        self._samples_since_reference = int(state.get("samples_since_reference", 0))
        self._cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self._awaiting_reset = bool(state.get("awaiting_reset", False))
        last_raw_score = state.get("last_raw_score")
        self._last_raw_score = None if last_raw_score is None else float(last_raw_score)
        self._last_valid_classes = tuple(
            int(value) for value in state.get("last_valid_classes", [])
        )
        self._trigger_count = int(state.get("trigger_count", 0))


@dataclass(frozen=True)
class ReplayBatch:
    features: torch.Tensor
    base_logits: torch.Tensor
    labels: torch.Tensor
    regime_ids: Optional[torch.Tensor] = None

    def __len__(self):
        return int(self.labels.numel())


class PersistentFeatureReplay:
    """Bounded, persistent feature replay with class-balanced sampling."""

    STATE_VERSION = 1
    VALID_SAMPLING = {"class_balanced", "uniform"}
    VALID_REPLACEMENT = {"fifo", "reservoir"}

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        *,
        logit_dim: int = 2,
        sampling: str = "class_balanced",
        replacement: str = "fifo",
        seed: int = 0,
        store_regime_ids: bool = False,
    ):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.logit_dim = int(logit_dim)
        self.sampling = str(sampling).lower()
        self.replacement = str(replacement).lower()
        self.store_regime_ids = bool(store_regime_ids)
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if self.feature_dim <= 0 or self.logit_dim <= 0:
            raise ValueError("feature_dim and logit_dim must be positive")
        if self.sampling not in self.VALID_SAMPLING:
            raise ValueError(f"unsupported replay sampling mode: {self.sampling}")
        if self.replacement not in self.VALID_REPLACEMENT:
            raise ValueError(f"unsupported replay replacement mode: {self.replacement}")
        self._features = torch.empty((self.capacity, self.feature_dim), dtype=torch.float32)
        self._base_logits = torch.empty((self.capacity, self.logit_dim), dtype=torch.float32)
        self._labels = torch.empty((self.capacity,), dtype=torch.long)
        self._regime_ids = (
            torch.empty((self.capacity,), dtype=torch.long)
            if self.store_regime_ids
            else None
        )
        self._size = 0
        self._write_pos = 0
        self._seen = 0
        self._generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._sample_calls = 0
        self._added_by_label = {0: 0, 1: 0}

    def __len__(self):
        return int(self._size)

    def clear(self):
        self._size = 0
        self._write_pos = 0

    def add(self, features, base_logits, labels, regime_id=None):
        features = _cpu_float_matrix(
            features,
            columns=self.feature_dim,
            name="features",
        )
        base_logits = _cpu_float_matrix(
            base_logits,
            columns=self.logit_dim,
            name="base_logits",
        )
        labels = _cpu_binary_labels(labels, rows=features.size(0))
        if base_logits.size(0) != features.size(0):
            raise ValueError("features and base_logits row counts must match")
        count = int(features.size(0))
        if count == 0:
            return
        if self.store_regime_ids:
            if regime_id is None:
                raise ValueError("regime_id is required when store_regime_ids=True")
            if torch.is_tensor(regime_id) or isinstance(regime_id, (list, tuple)):
                regime_ids = torch.as_tensor(regime_id, dtype=torch.long).reshape(-1)
                if regime_ids.numel() != count:
                    raise ValueError("regime_id tensor must match the batch row count")
            else:
                regime_ids = torch.full((count,), int(regime_id), dtype=torch.long)
        else:
            regime_ids = None

        for label in (0, 1):
            self._added_by_label[label] += int((labels == label).sum().item())
        if self.capacity == 0:
            self._seen += count
            return
        if self.replacement == "fifo":
            self._add_fifo(features, base_logits, labels, regime_ids)
        else:
            self._add_reservoir(features, base_logits, labels, regime_ids)

    def _write_rows(self, indices, features, base_logits, labels, regime_ids):
        indices = torch.as_tensor(indices, dtype=torch.long)
        self._features[indices] = features
        self._base_logits[indices] = base_logits
        self._labels[indices] = labels
        if self._regime_ids is not None and regime_ids is not None:
            self._regime_ids[indices] = regime_ids

    def _add_fifo(self, features, base_logits, labels, regime_ids):
        count = int(features.size(0))
        self._seen += count
        if count >= self.capacity:
            features = features[-self.capacity :]
            base_logits = base_logits[-self.capacity :]
            labels = labels[-self.capacity :]
            regime_ids = None if regime_ids is None else regime_ids[-self.capacity :]
            indices = torch.arange(self.capacity)
            self._write_rows(indices, features, base_logits, labels, regime_ids)
            self._size = self.capacity
            self._write_pos = 0
            return
        indices = (torch.arange(count) + self._write_pos) % self.capacity
        self._write_rows(indices, features, base_logits, labels, regime_ids)
        self._write_pos = int((self._write_pos + count) % self.capacity)
        self._size = min(self.capacity, self._size + count)

    def _add_reservoir(self, features, base_logits, labels, regime_ids):
        for row in range(int(features.size(0))):
            seen_index = self._seen
            self._seen += 1
            if seen_index < self.capacity:
                destination = seen_index
                self._size += 1
            else:
                draw = int(
                    torch.randint(
                        0,
                        seen_index + 1,
                        (1,),
                        generator=self._generator,
                    ).item()
                )
                if draw >= self.capacity:
                    continue
                destination = draw
            row_regime = None if regime_ids is None else regime_ids[row : row + 1]
            self._write_rows(
                [destination],
                features[row : row + 1],
                base_logits[row : row + 1],
                labels[row : row + 1],
                row_regime,
            )

    def _available_indices(self, regime_id=None) -> torch.Tensor:
        indices = torch.arange(self._size, dtype=torch.long)
        if regime_id is not None:
            if self._regime_ids is None:
                raise ValueError("regime-filtered replay requires store_regime_ids=True")
            indices = indices[self._regime_ids[: self._size] == int(regime_id)]
        return indices

    def _draw(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0 or indices.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        if indices.numel() >= count:
            order = torch.randperm(indices.numel(), generator=self._generator)[:count]
            return indices[order]
        draws = torch.randint(
            0,
            indices.numel(),
            (count,),
            generator=self._generator,
        )
        return indices[draws]

    def sample(self, n: int, device="cpu", regime_id=None) -> Optional[ReplayBatch]:
        n = int(n)
        if n <= 0 or self._size <= 0:
            return None
        available = self._available_indices(regime_id)
        if available.numel() == 0:
            return None
        sample_count = min(n, max(int(available.numel()), n if self.sampling == "class_balanced" else 0))
        if self.sampling == "class_balanced":
            zero_ids = available[self._labels[available] == 0]
            one_ids = available[self._labels[available] == 1]
            if zero_ids.numel() and one_ids.numel():
                zero_count = sample_count // 2
                one_count = sample_count - zero_count
                selected = torch.cat(
                    [self._draw(zero_ids, zero_count), self._draw(one_ids, one_count)]
                )
                selected = selected[
                    torch.randperm(selected.numel(), generator=self._generator)
                ]
            else:
                selected = self._draw(available, min(n, int(available.numel())))
        else:
            selected = self._draw(available, min(n, int(available.numel())))
        self._sample_calls += 1
        target_device = torch.device(device)
        regime_ids = (
            None
            if self._regime_ids is None
            else self._regime_ids[selected].to(target_device, non_blocking=True)
        )
        return ReplayBatch(
            self._features[selected].to(target_device, non_blocking=True),
            self._base_logits[selected].to(target_device, non_blocking=True),
            self._labels[selected].to(target_device, non_blocking=True),
            regime_ids,
        )

    def recent(self, n: int, device="cpu") -> Optional[ReplayBatch]:
        if self._size <= 0 or int(n) <= 0:
            return None
        if self.replacement != "fifo":
            raise ValueError("recent() is defined only for FIFO replay")
        count = min(int(n), self._size)
        if self._size < self.capacity:
            indices = torch.arange(self._size - count, self._size)
        else:
            oldest = self._write_pos
            chronological = (torch.arange(self._size) + oldest) % self.capacity
            indices = chronological[-count:]
        target_device = torch.device(device)
        regime_ids = (
            None
            if self._regime_ids is None
            else self._regime_ids[indices].to(target_device)
        )
        return ReplayBatch(
            self._features[indices].to(target_device),
            self._base_logits[indices].to(target_device),
            self._labels[indices].to(target_device),
            regime_ids,
        )

    def diagnostics(self) -> dict:
        labels = self._labels[: self._size]
        counts = {str(label): int((labels == label).sum().item()) for label in (0, 1)}
        bytes_used = self._size * (
            self.feature_dim * 4
            + self.logit_dim * 4
            + 8
            + (8 if self.store_regime_ids else 0)
        )
        return {
            "capacity": self.capacity,
            "size": int(self._size),
            "seen": int(self._seen),
            "sampling": self.sampling,
            "replacement": self.replacement,
            "label_counts": counts,
            "added_by_label": {
                str(label): int(count) for label, count in self._added_by_label.items()
            },
            "sample_calls": int(self._sample_calls),
            "bytes_used": int(bytes_used),
            "store_regime_ids": self.store_regime_ids,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "configuration": {
                "capacity": self.capacity,
                "feature_dim": self.feature_dim,
                "logit_dim": self.logit_dim,
                "sampling": self.sampling,
                "replacement": self.replacement,
                "store_regime_ids": self.store_regime_ids,
            },
            "features": self._features[: self._size].clone(),
            "base_logits": self._base_logits[: self._size].clone(),
            "labels": self._labels[: self._size].clone(),
            "regime_ids": (
                None if self._regime_ids is None else self._regime_ids[: self._size].clone()
            ),
            "size": int(self._size),
            "write_pos": int(self._write_pos),
            "seen": int(self._seen),
            "generator_state": self._generator.get_state(),
            "sample_calls": int(self._sample_calls),
            "added_by_label": dict(self._added_by_label),
        }

    def load_state_dict(self, state: dict):
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported replay state version")
        if state.get("configuration") != self.state_dict()["configuration"]:
            raise ValueError("replay state configuration does not match current configuration")
        size = int(state.get("size", 0))
        if not 0 <= size <= self.capacity:
            raise ValueError("serialized replay size exceeds configured capacity")
        features = _cpu_float_matrix(
            state.get("features", torch.empty((0, self.feature_dim))),
            columns=self.feature_dim,
            name="features",
        )
        base_logits = _cpu_float_matrix(
            state.get("base_logits", torch.empty((0, self.logit_dim))),
            columns=self.logit_dim,
            name="base_logits",
        )
        labels = _cpu_binary_labels(state.get("labels", torch.empty(0)), rows=size)
        if features.size(0) != size or base_logits.size(0) != size:
            raise ValueError("serialized replay tensors do not match replay size")
        self._features[:size] = features
        self._base_logits[:size] = base_logits
        self._labels[:size] = labels
        if self._regime_ids is not None:
            regime_ids = state.get("regime_ids")
            if not torch.is_tensor(regime_ids) or regime_ids.numel() != size:
                raise ValueError("serialized replay is missing regime ids")
            self._regime_ids[:size] = regime_ids.detach().cpu().long()
        self._size = size
        self._write_pos = int(state.get("write_pos", 0))
        self._seen = int(state.get("seen", size))
        self._generator.set_state(state["generator_state"].detach().cpu())
        self._sample_calls = int(state.get("sample_calls", 0))
        added = state.get("added_by_label", {})
        self._added_by_label = {
            0: int(added.get(0, added.get("0", 0))),
            1: int(added.get(1, added.get("1", 0))),
        }


@dataclass(frozen=True)
class AllocationEvent:
    sample_offset: int
    residual_expert_id: int
    total_regimes: int
    mode: str
    reason: str
    scheduled_offset: Optional[int] = None


class AllocationController:
    """Expert allocation policies, with detected mode free of stage metadata."""

    STATE_VERSION = 1
    VALID_MODES = {"detected", "single", "fixed", "random", "oracle", "none"}

    def __init__(
        self,
        mode: str = "detected",
        *,
        num_regimes: int = 0,
        total_online_samples: Optional[int] = None,
        positions: Optional[Sequence[int]] = None,
        seed: int = 0,
        max_residual_experts: int = 32,
    ):
        self.mode = str(mode).lower()
        self.num_regimes = int(num_regimes)
        self.total_online_samples = (
            None if total_online_samples is None else int(total_online_samples)
        )
        self.seed = int(seed)
        self.max_residual_experts = int(max_residual_experts)
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"unsupported allocation mode: {self.mode}")
        if self.max_residual_experts <= 0:
            raise ValueError("max_residual_experts must be positive")
        self._positions = self._resolve_positions(positions)
        self._next_position = 0
        self._residual_count = 0
        self._events: list[AllocationEvent] = []

    def _resolve_positions(self, positions):
        if self.mode not in {"fixed", "random"}:
            if positions:
                raise ValueError("positions are valid only for fixed/random allocation")
            return []
        target = max(self.num_regimes - 1, 0)
        if self.num_regimes < 1:
            raise ValueError("fixed/random allocation requires num_regimes >= 1")
        if target == 0:
            if positions not in (None, [], ()):
                raise ValueError("num_regimes=1 cannot have allocation positions")
            return []
        if positions is not None:
            resolved = sorted(int(value) for value in positions)
            if len(resolved) != target or len(set(resolved)) != target:
                raise ValueError("allocation positions must be unique and match num_regimes - 1")
            if resolved[0] < 0:
                raise ValueError("allocation positions must be non-negative")
            return resolved
        if self.total_online_samples is None or self.total_online_samples <= 0:
            raise ValueError("fixed/random allocation requires total_online_samples")
        if target > self.total_online_samples:
            raise ValueError("more residual allocations requested than online samples")
        if self.mode == "fixed":
            return [
                int(index * self.total_online_samples // target)
                for index in range(target)
            ]
        if target == 1:
            return [0]
        if self.total_online_samples <= 1:
            raise ValueError("random allocation has no non-zero sample positions")
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        remaining = torch.randperm(
            self.total_online_samples - 1,
            generator=generator,
        )[: target - 1]
        return [0] + sorted(int(value) + 1 for value in remaining.tolist())

    def _allocate(self, sample_offset: int, reason: str, scheduled_offset=None):
        if self._residual_count >= self.max_residual_experts:
            raise RuntimeError(
                "RACE reached max_residual_experts; refusing silent capacity growth"
            )
        self._residual_count += 1
        event = AllocationEvent(
            sample_offset=int(sample_offset),
            residual_expert_id=self._residual_count,
            total_regimes=self._residual_count + 1,
            mode=self.mode,
            reason=str(reason),
            scheduled_offset=(
                None if scheduled_offset is None else int(scheduled_offset)
            ),
        )
        self._events.append(event)
        return event

    def should_allocate(
        self,
        sample_offset: int,
        *,
        detected: bool = False,
        oracle: bool = False,
    ) -> Optional[AllocationEvent]:
        """Return at most one allocation event before the batch at ``sample_offset``."""

        sample_offset = int(sample_offset)
        if sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        if self.mode == "none":
            return None
        if self.mode == "detected":
            return self._allocate(sample_offset, "detector") if detected else None
        if self.mode == "single":
            if self._residual_count == 0:
                return self._allocate(sample_offset, "first_online_batch")
            return None
        if self.mode == "oracle":
            return self._allocate(sample_offset, "oracle_boundary") if oracle else None
        if self._next_position >= len(self._positions):
            return None
        scheduled = self._positions[self._next_position]
        if scheduled > sample_offset:
            return None
        self._next_position += 1
        return self._allocate(sample_offset, self.mode, scheduled_offset=scheduled)

    @property
    def residual_count(self):
        return int(self._residual_count)

    @property
    def positions(self):
        return tuple(self._positions)

    @property
    def events(self):
        return tuple(self._events)

    def diagnostics(self) -> dict:
        return {
            "mode": self.mode,
            "num_residual_experts": int(self._residual_count),
            "num_regimes": int(self._residual_count + 1),
            "configured_positions": list(self._positions),
            "events": [
                {
                    "sample_offset": event.sample_offset,
                    "residual_expert_id": event.residual_expert_id,
                    "total_regimes": event.total_regimes,
                    "mode": event.mode,
                    "reason": event.reason,
                    "scheduled_offset": event.scheduled_offset,
                }
                for event in self._events
            ],
            "uses_boundary_signal": self.mode == "oracle",
            "uses_future_horizon": self.mode in {"fixed", "random"},
        }

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "configuration": {
                "mode": self.mode,
                "num_regimes": self.num_regimes,
                "total_online_samples": self.total_online_samples,
                "seed": self.seed,
                "max_residual_experts": self.max_residual_experts,
                "positions": list(self._positions),
            },
            "next_position": int(self._next_position),
            "residual_count": int(self._residual_count),
            "events": list(self._events),
        }

    def load_state_dict(self, state: dict):
        if int(state.get("version", -1)) != self.STATE_VERSION:
            raise ValueError("unsupported allocation-controller state version")
        expected = self.state_dict()["configuration"]
        if state.get("configuration") != expected:
            raise ValueError("allocation-controller state configuration mismatch")
        self._next_position = int(state.get("next_position", 0))
        self._residual_count = int(state.get("residual_count", 0))
        events = []
        for payload in state.get("events", []):
            if isinstance(payload, AllocationEvent):
                events.append(payload)
            else:
                events.append(AllocationEvent(**payload))
        self._events = events


__all__ = [
    "AllocationController",
    "AllocationEvent",
    "ChangeEvent",
    "ClassConditionalChangeDetector",
    "DiagonalMoments",
    "PersistentFeatureReplay",
    "ReplayBatch",
    "symmetric_diagonal_gaussian_kl",
]
