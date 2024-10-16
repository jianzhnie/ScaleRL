import math
from typing import Callable, Optional, Union

import torch
import torch.optim as optim
from torch import Tensor


class SharedAdam(optim.Adam):
    """Implements the Adam optimization algorithm with shared states across
    processes. This is useful in multi-processing environments like A3C
    (Asynchronous Advantage Actor-Critic), where multiple workers share the
    optimizer states.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float): Weight decay (L2 penalty) (default: 0).
    """

    def __init__(
        self,
        params: Union[iter, dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initialize the SharedAdam optimizer with shared states (step,
        exp_avg, and exp_avg_sq)."""
        super(SharedAdam, self).__init__(params,
                                         lr=lr,
                                         betas=betas,
                                         eps=eps,
                                         weight_decay=weight_decay)

        # Initialize shared states (step, exp_avg, exp_avg_sq) for each parameter in the optimizer
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] = torch.zeros(1)
                    # Shared step counter
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient

    def share_memory(self) -> None:
        """Share the optimizer states (step, exp_avg, exp_avg_sq) across
        processes using shared memory.

        This is important for multi-processing scenarios.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'].share_memory_()
                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()

    def step(
        self,
        closure: Optional[Callable[[], Optional[Tensor]]] = None
    ) -> Optional[Tensor]:
        """Perform a single optimization step (parameter update).

        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            Optional[Tensor]: The loss returned by the closure, if any.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, consider SparseAdam instead.'
                    )

                state = self.state[p]

                # Get the exponential moving averages of gradient and squared gradient
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1  # Update step counter

                # Apply weight decay (L2 regularization) if applicable
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Update biased first moment estimate (exp_avg)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # Update biased second moment estimate (exp_avg_sq)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Compute denominator for bias-corrected step
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Compute bias-corrected learning rates
                bias_correction1 = 1 - beta1**state['step'].item()
                bias_correction2 = 1 - beta2**state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                # Update parameters
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
