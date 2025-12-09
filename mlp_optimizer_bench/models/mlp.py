import math
import torch
import torch.nn as nn
from typing import List, Callable, Optional, Literal


def spectral_mup_init_method_normal(sigma):
    """Spectral MuP initialization: σ * √(d_out/d_in) / ||W'||₂ * W'

    This initialization method applies spectral normalization and MuP scaling to linear layers.
    For 2D weight matrices W ∈ R^(d_out × d_in):
    1. Initialize W' ~ N(0, σ)
    2. Compute spectral norm s = ||W'||₂
    3. Apply scaling: W = σ * √(d_out/d_in) / s * W'

    Only applies to 2D linear layer weights. Skips:
    - Non-2D parameters (biases, layernorm, etc.)
    - lm_head (identified by d_out/d_in > 10, as vocab_size >> hidden_size)
    - Embedding layers (use separate embedding_init_method)

    Args:
        sigma: Standard deviation for the initial normal distribution.

    Returns:
        Initialization function that can be applied to tensors.
    """
    def init_(tensor):
        # Skip non-2D parameters (bias, layernorm, etc.)
        if len(tensor.shape) != 2:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        d_out, d_in = tensor.shape

        # Skip lm_head: identified by large output dimension ratio
        # vocab_size is typically >> hidden_size (e.g., 151936 >> 2048)
        if d_out > 50000 or d_in > 50000:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        # Step 1: Initialize W' ~ N(0, σ)
        torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        # Step 2: Compute spectral norm s = ||W'||₂
        spectral_norm = torch.linalg.matrix_norm(tensor, ord=2)

        # Step 3: Apply MuP scaling: W = σ * √(d_out/d_in) / s * W'
        mup_scale = math.sqrt(d_out / d_in) / spectral_norm
        tensor.data.mul_(mup_scale)

        return tensor

    return init_


class MLP(nn.Module):
    """
    通用多层感知机（MLP）

    参数：
        input_dim:  输入特征维度
        hidden_dims: 每一层隐藏层的维度列表，例如 [128, 64, 32]
        output_dim: 输出维度（例如分类 num_classes / 回归输出维度）
        activation: 隐藏层之间使用的激活函数（nn.Module 实例或类），默认 nn.ReLU
        activate_last: 是否在最后一层 Linear 后加激活函数（一般设为 False）
        init_type: 权重初始化方式，可自定义扩展
            - "xavier_uniform"
            - "xavier_normal"
            - "kaiming_uniform"
            - "kaiming_normal"
            - "normal"
            - "uniform"
            - "spectral_mup_init"
            - "none"（不做特殊初始化）
        init_gain: 用于部分初始化方法的 gain，例如 LeakyReLU 的负斜率等场景可用
        init_sigma: 用于 spectral_mup_init 的标准差参数
        bias_init: bias 初始化方式：
            - "zero"（全 0）
            - "normal"（正态分布）
            - "none"（不处理）
        init_fn: 自定义初始化函数，优先级高于 init_type。
                 函数签名为 init_fn(m: nn.Module) -> None
                 在构造函数内部会对 MLP 内所有子模块调用该函数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        activate_last: bool = False,
        init_type: Literal[
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "normal",
            "uniform",
            "spectral_mup_init",
            "none",
        ] = "xavier_uniform",
        init_gain: float = 1.0,
        init_sigma: float = 0.02,
        bias_init: Literal["zero", "normal", "none"] = "zero",
        init_fn: Optional[Callable[[nn.Module], None]] = None,
    ):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim, bias=False))

            # 最后一层是否加激活
            if i < len(dims) - 2 or activate_last:
                # activation 可以是类或者实例
                layers.append(activation() if isinstance(activation, type) else activation)

        self.net = nn.Sequential(*layers)

        # 初始化
        if init_fn is not None:
            # 用户自定义初始化函数优先
            self.net.apply(init_fn)
        else:
            self._apply_init(init_type, init_gain, init_sigma, bias_init)

    def _apply_init(
        self,
        init_type: str,
        gain: float,
        sigma: float,
        bias_init: str,
    ):
        if init_type == "none" and bias_init == "none":
            return

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 权重初始化
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif init_type == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02 * gain)
                elif init_type == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1 * gain, b=0.1 * gain)
                elif init_type == "spectral_mup_init":
                    init_fn = spectral_mup_init_method_normal(sigma)
                    init_fn(m.weight)
                elif init_type == "none":
                    pass
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")

                # bias 初始化
                if m.bias is not None:
                    if bias_init == "zero":
                        nn.init.zeros_(m.bias)
                    elif bias_init == "normal":
                        nn.init.normal_(m.bias, mean=0.0, std=0.02)
                    elif bias_init == "none":
                        pass
                    else:
                        raise ValueError(f"Unknown bias_init: {bias_init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)