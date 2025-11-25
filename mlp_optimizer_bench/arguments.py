import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # ===== 数据集相关 =====
    parser.add_argument(
        "--dataset",
        type=str,
        default="high_order_cls",
        choices=["high_order_cls", "function_reg"],
        help="选择数据集：high_order_cls(高阶分类) 或 function_reg(函数回归)",
    )

    # 通用样本、特征设置（主要给 high_order_cls 用）
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-features", type=int, default=50)

    # function_reg 专用参数
    parser.add_argument(
        "--func-task-type",
        type=str,
        default="piecewise_sin",
        choices=["piecewise_sin", "highdim_combo"],
        help="function_reg 任务类型",
    )
    parser.add_argument("--func-n-samples", type=int, default=5000)
    parser.add_argument("--func-input-dim", type=int, default=20)

    # ===== DataLoader 相关 =====
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    # ===== 模型相关 =====
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,256",
        help="MLP 隐藏层维度，以逗号分隔，例如 '256,256,128'",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh"],
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="xavier_uniform",
        choices=[
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "normal",
            "uniform",
            "none",
        ],
    )
    parser.add_argument(
        "--bias-init",
        type=str,
        default="zero",
        choices=["zero", "normal", "none"],
    )

    # ===== 训练控制 =====
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="训练的最大步数（以 optimizer.step() 的次数为单位）",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="如果在达到 max_steps 前先达到 max_epochs，则也会停止；一般给一个较大上限即可。",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="每多少个 step 在验证集上评估一次（<=0 表示不评估）",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="每多少个 step 打印一次训练日志",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu（例如 'cuda:0'）",
    )

    # ===== 优化器相关 =====
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "muon"],
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Muon 专用
    parser.add_argument("--muon-momentum-beta", type=float, default=0.95)
    parser.add_argument("--muon-use-nesterov", action="store_true")
    parser.add_argument(
        "--muon-scale-mode",
        type=str,
        default="align_adamw_rms",
        choices=["align_adamw_rms", "shape_scaling", "spectral_mup"],
    )
    parser.add_argument("--muon-extra-scale-factor", type=float, default=1.0)
    parser.add_argument("--muon-num-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-coefficient-type",
        type=str,
        default="quintic",
        choices=["simple", "quintic", "polar_express"],
    )
    parser.add_argument("--muon-use-syrk", action="store_true")

    # ... 你可以在这里继续添加其他超参数

    args = parser.parse_args()

    # 解析 hidden-dims 字符串为列表
    if isinstance(args.hidden_dims, str):
        args.hidden_dims = [
            int(x) for x in args.hidden_dims.split(",") if x.strip() != ""
        ]

    return args