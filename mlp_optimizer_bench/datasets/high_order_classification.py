# mlp_optimizer_bench/datasets/high_order_classification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader

try:
    from sklearn.datasets import make_classification
except ImportError as e:
    raise ImportError(
        "high_order_classification requires scikit-learn.\n"
        "Please install it with `pip install scikit-learn`."
    ) from e

from .utils import (
    StandardScalerWrapper,
    split_train_val_test_indices,
)


@dataclass
class HighOrderClassificationConfig:
    n_samples: int = 10000
    n_features: int = 50
    n_informative: int = 10
    n_redundant: int = 10
    n_repeated: int = 0
    n_classes: int = 5
    n_clusters_per_class: int = 2
    class_sep: float = 1.0
    flip_y: float = 0.01
    random_state: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    standardize: bool = True


class HighOrderClassificationDataset(Dataset):
    """高阶非线性多分类数据集（基于 sklearn.make_classification）。

    返回:
        x: float32 tensor, shape (n_features,)
        y: long tensor, 单个标量类别标签 [0, n_classes-1]
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        standardizer: Optional[StandardScalerWrapper] = None,
        device: Optional[torch.device] = None,
    ):
        if standardizer is not None:
            X = standardizer.transform(X)

        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def generate_high_order_classification(
    config: Optional[HighOrderClassificationConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """生成高阶非线性多分类任务数据，并切分 train/val/test。

    返回:
        {
            "train": Dataset,
            "val": Dataset,
            "test": Dataset,
            "meta": {...}
        }
    """
    if config is None:
        config = HighOrderClassificationConfig()

    X, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=config.n_redundant,
        n_repeated=config.n_repeated,
        n_classes=config.n_classes,
        n_clusters_per_class=config.n_clusters_per_class,
        class_sep=config.class_sep,
        flip_y=config.flip_y,
        random_state=config.random_state,
    )

    # 标准化（只在训练集上 fit）
    scaler = None
    if config.standardize:
        scaler = StandardScalerWrapper()
        scaler.fit(X)

    idx_train, idx_val, idx_test = split_train_val_test_indices(
        n_samples=config.n_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state,
    )

    def _subset_dataset(indices: np.ndarray) -> HighOrderClassificationDataset:
        return HighOrderClassificationDataset(
            X[indices],
            y[indices],
            standardizer=scaler,
            device=device,
        )

    train_ds = _subset_dataset(idx_train)
    val_ds = _subset_dataset(idx_val)
    test_ds = _subset_dataset(idx_test)

    meta = {
        "n_features": config.n_features,
        "n_classes": config.n_classes,
        "config": config,
    }

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "meta": meta,
    }


def get_high_order_classification_dataloaders(
    batch_size: int = 128,
    config: Optional[HighOrderClassificationConfig] = None,
    device: Optional[torch.device] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """便捷函数：直接返回 DataLoader。"""
    data = generate_high_order_classification(config=config, device=device)
    train_loader = DataLoader(
        data["train"], batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    val_loader = DataLoader(
        data["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        data["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, data["meta"]