# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable


# TODO(@boxiangw): remove this once bump to python 3.12
try:
    from typing import override
except ImportError:
    from typing_extensions import override

import torch
import torch.optim as optim
from absl import logging
from torch.optim.optimizer import ParamsT

from mlp_optimizer_bench.optimizers import mixin as opt_mixin
from mlp_optimizer_bench.optimizers import utils


_args_doc = """params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: The learning rate used by the internal SGD.
        momentum_beta: The momentum used by the internal SGD.
        weight_decay: The weight decay used by the optimizer, default to be decoupled weight decay.
            See Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        use_nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        weight_decay_method: Method to apply weight decay, see :class:`~emerging_optimizers.mixin.WeightDecayMixin`
            for more details.
        fp32_matmul_prec: Precision of the matmul operations in optimizer states GEMM operations.
        adamw_betas: The betas for AdamW optimizer used for 1D parameters. Default: (0.9, 0.95)
        adamw_eps: The epsilon for AdamW optimizer used for 1D parameters. Default: 1e-8
"""


class OrthogonalizedOptimizer(opt_mixin.WeightDecayMixin, optim.Optimizer):
    """Base class for orthogonalized optimizers.

    This class is a wrapper around a base optimizer that performs orthogonalization on the updates.
    The theoretical foundation of orthogonalization for stochastic gradient descent was developed by the
    following papers:

    - Carlson, D., Cevher, V., and Carin, L. *Stochastic spectral descent for Restricted Boltzmann Machines.*
      In International Conference on Artificial Intelligence and Statistics (2015a).
    - Carlson, D., Hsieh, Y.-P., Collins, E., Carin, L., and Cevher, V.
      *Stochastic Spectral Descent for Discrete Graphical Models.*
      In IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 2, pp. 296-311 (2016).
    - Carlson, D., Collins, E., Hsieh, Y.-P., Carin, L., and Cevher, V.
      *Preconditioned spectral descent for deep learning.*
      In Neural Information Processing Systems (2015b).
    - Flynn, T. *The duality structure gradient descent algorithm: analysis and applications to neural networks.*
      arXiv preprint arXiv:1708.00523 (2017). [`arXiv:1708.00523 <https://arxiv.org/abs/1708.00523>`_]

    Note:
        OrthogonalizedOptimizer as base class doesn't directly support orthogonalizing fused parameters separately.
        Subclass can override the orthogonalize function to support this, see example below.

    .. code-block:: python
       :caption: Split QKV example

       class SplitQkvOrthogonalizedOptimizer(OrthogonalizedOptimizer):
           def __init__(..., split_qkv_shapes):
               super().__init__(...)
               self.qkv_split_shapes = split_qkv_shapes

           def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:

               # Alternative is passing "is_qkv" to scaled_orthogonalize_fn and split inside the
               # scaled_orthogonalize_fn.
               if getattr(p, "is_qkv", False) or kwargs.get("is_qkv", False):
                   qkv_grads = torch.split(grad, self.qkv_split_shapes, dim=0)
                   qkv_orthogonalized = [self.scaled_orthogonalize_fn(g) for g in qkv_grads]
                   grad = torch.cat([orthogonalized for orthogonalized in qkv_orthogonalized])
               else:
                   grad = self.scaled_orthogonalize_fn(grad)

               return grad

    Args:
        {_args_doc}
        scaled_orthogonalize_fn: Function to orthogonalize and scale the updates.
        **kwargs: Arguments passed through to the base optimizer.

    Note:
        Keyword arguments passed through are not checked here. Optimizer inherited from this class should check them.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        momentum_beta: float,
        weight_decay: float,
        *,
        use_nesterov: bool,
        weight_decay_method: opt_mixin.WeightDecayT,
        fp32_matmul_prec: str,
        scaled_orthogonalize_fn: Callable | None = None,
        log_per_module_update_rms: bool = False,
        log_per_module_grad_rms: bool = False,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        **kwargs: Any,
    ):
        if scaled_orthogonalize_fn is None:
            logging.warning("scaled_orthogonalize_fn not provided. Using noop")
            scaled_orthogonalize_fn = torch.nn.Identity()

        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov
        self.weight_decay_method = weight_decay_method
        self.log_per_module_update_rms = log_per_module_update_rms
        self.per_module_update_rms = {}
        self.log_per_module_grad_rms = log_per_module_grad_rms
        self.per_module_grad_rms = {}

        default_args_dict = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            **kwargs,
        )

        super().__init__(params, default_args_dict)
        self.scaled_orthogonalize_fn = scaled_orthogonalize_fn
        self.global_step = 0  # For AdamW bias correction

    @torch.no_grad()  # type: ignore[misc]
    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        # Increment global step counter for AdamW
        self.global_step += 1

        # Clear previous update RMS statistics (once per optimizer step)
        if self.log_per_module_update_rms:
            self.per_module_update_rms.clear()
        # Clear previous grad RMS statistics (once per optimizer step)
        if self.log_per_module_grad_rms:
            self.per_module_grad_rms.clear()

        # ========= Phase 1: Handle 2D+ parameters with orthogonalization =========
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() < 2:  # Skip 1D parameters in this phase
                    continue

                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                # initialize momentum buffer
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                # Subsequent update to exp_avg are all inplace, so it is not assigned back to state.
                exp_avg = state["momentum_buffer"]

                self._apply_weight_decay_inplace(
                    p,
                    grad,
                    group["lr"],
                    group["weight_decay"],
                )

                # update momentum buffer with EMA of gradient
                exp_avg.lerp_(grad, 1 - group["momentum_beta"])

                # include nesterov momentum
                if self.use_nesterov:
                    grad = grad.lerp(exp_avg, group["momentum_beta"])
                else:
                    grad = exp_avg

                # Compute grad RMS if logging is enabled (on the momentum/Nesterov-adjusted grad)
                if self.log_per_module_grad_rms:
                    grad_rms = torch.sqrt(torch.mean(grad ** 2)).item()
                    param_name = getattr(p, 'param_name', None)
                    if param_name:
                        self.per_module_grad_rms[param_name] = grad_rms

                with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                    group_kwargs = {k: v for k, v in group.items() if k != "params"}
                    grad = self.orthogonalize(p, grad, **group_kwargs)

                # Compute update RMS if logging is enabled
                # Note: update RMS is computed on the update direction BEFORE multiplying by lr
                # and does NOT include weight decay (which is applied directly to params in decoupled mode)
                if self.log_per_module_update_rms:
                    update_rms = torch.sqrt(torch.mean(grad ** 2)).item()

                    # Get full parameter name (including .weight/.bias)
                    param_name = getattr(p, 'param_name', None)
                    if param_name == None:
                        print("No params name!")
                    if param_name:
                        self.per_module_update_rms[param_name] = update_rms

                # perform weight update
                # scale is applied to have update RMS == 1
                p.add_(grad, alpha=-group["lr"])

        # ========= Phase 2: Handle 1D parameters with AdamW =========
        for group in self.param_groups:
            beta1, beta2 = group.get('adamw_betas', (0.9, 0.95))
            eps = group.get('adamw_eps', 1e-8)

            for p in group["params"]:
                if p.dim() >= 2:  # Skip 2D+ parameters in this phase
                    continue

                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]

                # Initialize AdamW buffers
                if 'adamw_exp_avg' not in state:
                    state['adamw_exp_avg'] = torch.zeros_like(grad)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(grad)

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']

                # Update biased first moment estimate
                buf1.lerp_(grad, 1 - beta1)
                # Update biased second raw moment estimate
                buf2.lerp_(grad.square(), 1 - beta2)

                # Compute bias-corrected update direction
                g = buf1 / (eps + buf2.sqrt())

                # Bias correction
                bias_correction1 = 1 - beta1 ** self.global_step
                bias_correction2 = 1 - beta2 ** self.global_step
                scale = bias_correction1 / bias_correction2 ** 0.5

                # Apply decoupled weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Apply update
                p.data.add_(g, alpha=-group["lr"] / scale)

                # Log grad/update RMS if enabled
                if self.log_per_module_grad_rms:
                    grad_rms = torch.sqrt(torch.mean(grad ** 2)).item()
                    param_name = getattr(p, 'param_name', None)
                    if param_name:
                        self.per_module_grad_rms[param_name] = grad_rms

                if self.log_per_module_update_rms:
                    update_rms = torch.sqrt(torch.mean(g ** 2)).item()
                    param_name = getattr(p, 'param_name', None)
                    if param_name:
                        self.per_module_update_rms[param_name] = update_rms

        return loss

    def get_update_rms_dict(self):
        """Get per-module update RMS statistics.

        Returns:
            Dictionary mapping module names to their update RMS values, or None if logging is disabled.
        """
        return self.per_module_update_rms if self.log_per_module_update_rms else None

    def get_grad_rms_dict(self):
        """Get per-module grad RMS statistics.

        Returns:
            Dictionary mapping module names to their grad RMS values, or None if logging is disabled.
        """
        return self.per_module_grad_rms if self.log_per_module_grad_rms else None

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        The default orthogonalize function calls the scaled_orthogonalize_fn with the gradient. Subclass can
        override this function to implement different orthogonalization logic as well as split fused parameters.
        For example, a scaled_orthogonalize_fn function can get attributes from p or from kwargs to determine if
        the parameter is a fused parameter and should be split for preconditioning.

        Args:
            p: The parameter tensor. It is necessary to pass param tensor in addition to momentum because a lot of
                information is only available in the param tensor, attributes for example. Although not used in
                this default orthogonalize function.
            grad: The momentum tensor.
            **kwargs: keyword arguments of the param_group that p was belonged to.

        Returns:
            The orthogonalized gradient tensor.
        """
        grad = self.scaled_orthogonalize_fn(grad)
        return grad


OrthogonalizedOptimizer.__doc__ = OrthogonalizedOptimizer.__doc__.format(_args_doc=_args_doc)  # type: ignore[union-attr]
