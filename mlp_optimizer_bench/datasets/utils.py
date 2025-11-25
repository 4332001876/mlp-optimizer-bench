# mlp_optimizer_bench/datasets/utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class StandardScalerWrapper:
    """简单版标准化器：z = (x - mean) / std"""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "StandardScalerWrapper":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # 避免除 0
        self.scale_[self.scale_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerWrapper must be fitted before transform.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def split_train_val_test_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机划分索引为 train/val/test。

    保证 train_ratio + val_ratio + test_ratio ≈ 1。
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, "
            f"got {train_ratio + val_ratio + test_ratio}"
        )

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    idx_train = indices[:n_train]
    idx_val = indices[n_train : n_train + n_val]
    idx_test = indices[n_train + n_val :]

    return idx_train, idx_val, idx_test