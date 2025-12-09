# run_optim_bench.py

from __future__ import annotations

import torch

from mlp_optimizer_bench.datasets import (
    HighOrderClassificationConfig,
    get_high_order_classification_dataloaders,
)
from mlp_optimizer_bench.datasets.function_regression import (
    FunctionRegressionConfig,
    get_function_regression_dataloaders,
)

from mlp_optimizer_bench.arguments import parse_args
from mlp_optimizer_bench.models import MLP
from mlp_optimizer_bench.training import Trainer, TrainerConfig
from mlp_optimizer_bench.optimizers.muon import Muon
from mlp_optimizer_bench.optimizers.spectral_ball import SpectralBall


def build_activation(name: str):
    name = name.lower()
    import torch.nn as nn

    if name == "relu":
        return nn.ReLU
    elif name == "gelu":
        return nn.GELU
    elif name == "tanh":
        return nn.Tanh
    else:
        raise ValueError(f"Unknown activation: {name}")


def build_optimizer(args, model_params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "muon":
        return Muon(
            model_params,
            lr=args.lr,
            momentum_beta=args.muon_momentum_beta,
            weight_decay=args.weight_decay,
            use_nesterov=args.muon_use_nesterov,
            coefficient_type=args.muon_coefficient_type,
            num_ns_steps=args.muon_num_ns_steps,
            scale_mode=args.muon_scale_mode,
            extra_scale_factor=args.muon_extra_scale_factor,
            use_syrk=args.muon_use_syrk,
        )
    elif args.optimizer == "spectral_ball":
        return SpectralBall(
            model_params,
            lr=args.lr,
            momentum_beta=args.spectral_ball_momentum,
            weight_decay=args.weight_decay,
            use_nesterov=args.spectral_ball_use_nesterov,
            power_iteration_steps=args.spectral_ball_power_iteration_steps,
            msign_steps=args.spectral_ball_msign_steps,
            solver=args.spectral_ball_solver,
            solver_tolerance_f=args.spectral_ball_solver_tolerance_f,
            solver_max_iterations=args.spectral_ball_solver_max_iterations,
            radius_mode=args.spectral_ball_radius_mode,
            scale_mode=args.spectral_ball_scale_mode,
            retract_mode=args.spectral_ball_retract_mode,
            retract_alpha=args.spectral_ball_retract_alpha,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========= 构造数据集 =========
    if args.dataset == "high_order_cls":
        cfg = HighOrderClassificationConfig(
            n_samples=args.n_samples,
            n_features=args.n_features,
            # 也可以把其他字段从 args 里读出来
        )
        train_loader, val_loader, test_loader, meta = (
            get_high_order_classification_dataloaders(
                batch_size=args.batch_size,
                config=cfg,
            )
        )
        input_dim = meta["n_features"]
        n_classes = meta["n_classes"]
        task_type = "classification"

    elif args.dataset == "function_reg":
        func_cfg = FunctionRegressionConfig(
            n_samples=args.func_n_samples,
            task_type=args.func_task_type,
            input_dim=args.func_input_dim,
        )
        train_loader, val_loader, test_loader, meta = (
            get_function_regression_dataloaders(
                batch_size=args.batch_size,
                config=func_cfg,
                device=device,
            )
        )
        input_dim = meta["input_dim"]
        n_classes = 1  # 回归输出维度为 1
        task_type = "regression"

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ========= 构建 MLP =========
    activation_cls = build_activation(args.activation)

    model = MLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=n_classes,
        activation=activation_cls,
        activate_last=False,
        init_type=args.init_type,
        init_sigma=args.init_sigma,
        bias_init=args.bias_init,
    )

    # ========= 构建优化器 =========
    optimizer = build_optimizer(args, model.parameters())

    # ========= Trainer =========
    trainer_cfg = TrainerConfig(
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        log_every=args.log_every,
        eval_every=args.eval_every,
        device=str(device),
        task_type=task_type,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
    )

    trainer.train()

    # 需要的话，可以在这里对 test_loader 做最终评估
    # （略）。


if __name__ == "__main__":
    main()