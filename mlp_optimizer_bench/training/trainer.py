# mlp_optimizer_bench/trainer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainerConfig:
    max_steps: int = 1000              # 训练总步数（以 optimizer.step 为单位）
    max_epochs: int = 1000             # 训练的最大 epoch 数上限
    log_every: int = 50                # 日志打印间隔（step）
    eval_every: int = 100              # 验证间隔（step，<=0 表示不评估）
    device: str = "cuda"
    task_type: Literal["classification", "regression"] = "classification"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config

        device = torch.device(config.device)
        self.model.to(device)
        self.device = device

        if config.task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif config.task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type in TrainerConfig: {config.task_type}")

    def _move_batch_to_device(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in self.val_loader:
            x, y = self._move_batch_to_device(batch)
            out = self.model(x)

            if self.cfg.task_type == "classification":
                loss = self.criterion(out, y.long())
                preds = out.argmax(dim=-1)
                total_correct += (preds == y.long()).sum().item()
                total_count += y.numel()
            else:
                # 回归
                loss = self.criterion(out, y)
                total_count += y.numel()

            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / max(total_count, 1)
        metrics = {"val_loss": avg_loss}

        if self.cfg.task_type == "classification" and total_count > 0:
            metrics["val_acc"] = total_correct / total_count

        return metrics

    def train(self):
        step = 0
        epoch = 0

        while step < self.cfg.max_steps and epoch < self.cfg.max_epochs:
            epoch += 1
            self.model.train()

            for batch in self.train_loader:
                if step >= self.cfg.max_steps:
                    break

                step += 1
                x, y = self._move_batch_to_device(batch)

                self.optimizer.zero_grad()
                out = self.model(x)

                if self.cfg.task_type == "classification":
                    loss = self.criterion(out, y.long())
                else:
                    loss = self.criterion(out, y)

                loss.backward()
                self.optimizer.step()

                # 日志
                if self.cfg.log_every > 0 and step % self.cfg.log_every == 0:
                    if self.cfg.task_type == "classification":
                        with torch.no_grad():
                            preds = out.argmax(dim=-1)
                            acc = (preds == y.long()).float().mean().item()
                        print(
                            f"[Step {step}] loss = {loss.item():.4f}, acc = {acc:.4f}"
                        )
                    else:
                        print(f"[Step {step}] loss = {loss.item():.4f}")

                # 验证
                if (
                    self.cfg.eval_every > 0
                    and self.val_loader is not None
                    and step % self.cfg.eval_every == 0
                ):
                    metrics = self.evaluate()
                    if metrics:
                        msg = " | ".join(
                            f"{k}={v:.4f}" for k, v in metrics.items()
                        )
                        print(f"[Eval @ step {step}] {msg}")

        print(f"Training finished: step={step}, epoch={epoch}")