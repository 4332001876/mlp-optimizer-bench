# mlp_optimizer_bench/datasets/sequence_parity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import split_train_val_test_indices


@dataclass
class SequenceParityConfig:
    n_samples: int = 10000
    seq_len: int = 20
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    as_float: bool = True   # True: 输入为 {0.0, 1.0} float; False: {0,1} long


class SequenceParityDataset(Dataset):
    """奇偶校验序列任务。

    输入:
        x: shape (seq_len,) 含 0/1
    标签:
        y: 0/1, 表示 1 的个数 mod 2
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        as_float: bool = True,
        device: Optional[torch.device] = None,
    ):
        if as_float:
            self.x = torch.from_numpy(X.astype(np.float32))
        else:
            self.x = torch.from_numpy(X.astype(np.int64))

        self.y = torch.from_numpy(y.astype(np.int64))

        if device is not None:
            self.x = self.x.to(device)
            self.y = self.y.to(device)

        self.as_float = as_float

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def generate_sequence_parity(
    config: Optional[SequenceParityConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if config is None:
        config = SequenceParityConfig()

    rng = np.random.RandomState(config.random_state)

    # X: (N, L) in {0, 1}
    X = rng.randint(0, 2, size=(config.n_samples, config.seq_len))

    # y: 奇偶校验 = sum(x) % 2
    y = X.sum(axis=1) % 2

    idx_train, idx_val, idx_test = split_train_val_test_indices(
        n_samples=config.n_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_state,
    )

    def _subset(indices: np.ndarray) -> SequenceParityDataset:
        return SequenceParityDataset(
            X[indices],
            y[indices],
            as_float=config.as_float,
            device=device,
        )

    train_ds = _subset(idx_train)
    val_ds = _subset(idx_val)
    test_ds = _subset(idx_test)

    meta = {
        "seq_len": config.seq_len,
        "input_dim": config.seq_len,  # MLP 输入维度 = 序列长度
        "n_classes": 2,
        "config": config,
    }

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "meta": meta,
    }


def get_sequence_parity_dataloaders(
    batch_size: int = 128,
    config: Optional[SequenceParityConfig] = None,
    device: Optional[torch.device] = None,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    data = generate_sequence_parity(config=config, device=device)
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