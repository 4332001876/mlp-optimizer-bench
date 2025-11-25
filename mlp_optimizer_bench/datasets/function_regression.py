# mlp_optimizer_bench/datasets/function_regression.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import (
    StandardScalerWrapper,
    split_train_val_test_indices,
)


@dataclass
class FunctionRegressionConfig:
    # 通用
    n_samples: int = 5000
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    standardize_x: bool = True
    standardize_y: bool = False  # 对于回归，通常保留真实尺度，更直观

    # 任务选择
    task_type: str = "piecewise_sin"  # "piecewise_sin" | "highdim_combo"

    # 对 piecewise_sin 的控制
    x_low: float = -2.0
    x_high: float = 2.0
    noise_std: float = 0.1

    # 对 highdim_combo 的控制
    input_dim: int = 20
    highdim_noise_std: float = 0.1


def piecewise_sin_function(x: np.ndarray, noise_std: float, rng: np.random.RandomState) -> np.ndarray:
    """分段高频/低频的正弦/余弦函数 + 噪声。

    x: shape (N, 1)
    return: shape (N, 1)
    """
    x_flat = x[:, 0]
    noise = noise_std * rng.randn(*x_flat.shape)
    y = np.where(
        x_flat < 0.0,
        np.sin(10.0 * x_flat) + noise,   # 左：高频 + 噪声
        np.cos(3.0 * x_flat) + noise,    # 右：低频 + 噪声
    )
    return y.reshape(-1, 1)


def highdim_combo_function(x: np.ndarray, noise_std: float, rng: np.random.RandomState) -> np.ndarray:
    """高维组合函数示例。

    x: shape (N, D)
    return: shape (N, 1)
    """
    # 为了稳妥，先判断维度是否足够
    D = x.shape[1]
    if D < 5:
        raise ValueError("highdim_combo_function expects input_dim >= 5")

    # 构造一些非线性组合
    y = (
        np.sin(x[:, 0])
        + 0.5 * x[:, 1] ** 2
        - 0.3 * np.log(np.abs(x[:, 2]) + 1.0)
        + 0.2 * x[:, 3] * x[:, 4]
    )

    # 若维度更多，可以再加点项
    if D > 5:
        y += 0.1 * np.sum(np.tanh(x[:, 5:10]), axis=1)

    noise = noise_std * rng.randn(x.shape[0])
    y = y + noise
    return y.reshape(-1, 1)


class FunctionRegressionDataset(Dataset):
    """函数回归数据集。

    返回:
        x: float32 tensor, shape (input_dim,)
        y: float32 tensor, shape (1,)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_scaler: Optional[StandardScalerWrapper] = None,
        y_scaler: Optional[StandardScalerWrapper] = None,
        device: Optional[torch.device] = None,
    ):
        if x_scaler is not None:
            X = x_scaler.transform(X)
        if y_scaler is not None:
            y = y_scaler.transform(y)

        self.x = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

        self._x_scaler = x_scaler
        self._y_scaler = y_scaler

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    @property
    def x_scaler(self) -> Optional[StandardScalerWrapper]:
        return self._x_scaler

    @property
    def y_scaler(self) -> Optional[StandardScalerWrapper]:
        return self._y_scaler


def generate_function_regression(
    config: Optional[FunctionRegressionConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if config is None:
        config = FunctionRegressionConfig()

    rng = np.random.RandomState(config.random_state)

    if config.task_type == "piecewise_sin":
        X = rng.uniform(
            low=config.x_low,
            high=config.x_high,
            size=(config.n_samples, 1),
        )
        y = piecewise_sin_function(X, noise_std=config.noise_std, rng=rng)
    elif config.task_type == "highdim_combo":
        X = rng.randn(config.n_samples, config.input_dim)
        y = highdim_combo_function(X, noise_std=config.highdim_noise_std, rng=rng)
    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")

    # 划分索引
    idx_train, idx_val, idx_test = split_train_val_test_indices(
        n_samples=config.n_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state,
    )

    # 标准化：只在训练集上 fit
    x_scaler = None
    y_scaler = None
    if config.standardize_x:
        x_scaler = StandardScalerWrapper()
        x_scaler.fit(X[idx_train])

    if config.standardize_y:
        y_scaler = StandardScalerWrapper()
        y_scaler.fit(y[idx_train])

    def _subset(indices: np.ndarray) -> FunctionRegressionDataset:
        return FunctionRegressionDataset(
            X[indices],
            y[indices],
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            device=device,
        )

    train_ds = _subset(idx_train)
    val_ds = _subset(idx_val)
    test_ds = _subset(idx_test)

    meta = {
        "input_dim": X.shape[1],
        "task_type": config.task_type,
        "config": config,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }

    return {
        "train": train_ds,
        "val": train_ds if val_ds is None else val_ds,
        "test": test_ds,
        "meta": meta,
    }


def get_function_regression_dataloaders(
    batch_size: int = 128,
    config: Optional[FunctionRegressionConfig] = None,
    device: Optional[torch.device] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    data = generate_function_regression(config=config, device=device)
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