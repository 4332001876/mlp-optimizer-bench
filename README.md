# MLP Optimizer Benchmark
```
mlp_optimizer_bench/
├── README.md                  # 项目说明、使用方法
├── requirements.txt           # 依赖（可选）
├── setup.py                   # 如果要做成可安装包，可选
├── mlp_optimizer_bench/
│   ├── __init__.py
│   ├── config.py             # 一些全局配置/超参数默认值
│   │
│   ├── datasets/             # 数据相关：生成或加载
│   │   ├── __init__.py
│   │   ├── toy_classification.py   # 简单二维数据集/线性可分数据等
│   │   ├── mnist.py                # （可选）加载MNIST之类
│   │   └── utils.py                # 数据归一化、拆分train/val等
│   │
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   └── mlp.py            # 简单 MLP（支持不同层数/激活）
│   │
│   ├── optimizers/           # 自己实现/封装的优化器
│   │   ├── __init__.py
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   ├── adam.py
│   │   └── base.py           # Optimizer 抽象基类
│   │
│   ├── training/             # 训练和评估逻辑
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练循环
│   │   └── metrics.py        # 记录 loss, accuracy, 收敛速度等
│   │
│   ├── benchmarks/           # 运行不同实验组合的脚本
│   │   ├── __init__.py
│   │   └── run_optim_bench.py    # 核心benchmark：对比各optimizer
│   │
│   └── utils/                # 通用小工具
│       ├── __init__.py
│       ├── seed.py           # 固定随机种子
│       └── plotting.py       # （可选）画loss曲线等
│
└── scripts/
    ├── run_toy.sh            # 方便一条命令跑toy实验
    └── run_mnist.sh          # （可选）跑较大数据
```