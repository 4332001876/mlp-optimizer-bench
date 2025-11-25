import argparse
from mlp_optimizer_bench.datasets import (
    HighOrderClassificationConfig,
    get_high_order_classification_dataloaders,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="high_order_cls")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    # ... 其他超参数
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset == "high_order_cls":
        cfg = HighOrderClassificationConfig(
            n_samples=args.n_samples,
            n_features=args.n_features,
            # 也可以把其他字段从 args 里读出来
        )
        train_loader, val_loader, test_loader, meta = \
            get_high_order_classification_dataloaders(
                batch_size=args.batch_size,
                config=cfg,
            )
        input_dim = meta["n_features"]
        n_classes = meta["n_classes"]
        # 构建 MLP，开始训练...

if __name__ == "__main__":
    main()