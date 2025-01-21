import torch
from torch.optim import Optimizer

class MomentumOptimizer(Optimizer):
    """
    Implements a simplified momentum-based optimizer with the same API as the original.
    """

    def __init__(self, params, lr=0.01, momentum=0.99):
        """
        Initializes the optimizer.

        Args:
            params (iterable): An iterable of parameters to optimize.
            lr (float, optional): Base learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.99.
            weight_decay (float, optional): Weight decay factor (L2 regularization). Defaults to 0.0.
            warmup_steps (int, optional): Number of steps for learning rate warm-up. Defaults to 0.
        """
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)
        self.prev_grad = None

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value if `closure` is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                gradient = p.grad.data

                if self.prev_grad is None:
                    self.prev_grad = torch.zeros_like(gradient)

                # Compute the momentum update
                momentum_update = gradient + group['momentum'] * self.prev_grad
                self.prev_grad = gradient.clone().detach()

                # Update the parameters
                p.data.add_(- group['lr'], momentum_update)

        return loss

    def reset(self):
        """
        Resets the optimizer's internal state (momentum).
        """
        self.prev_grad = None

    def set_lr(self, new_lr):
        """
        Dynamically updates the learning rate for all parameter groups.

        Args:
            new_lr (float): The new learning rate value to set.
        """
        for group in self.param_groups:
            group['lr'] = new_lr

    def get_lr(self):
        """
        Retrieves the current learning rates for all parameter groups.

        Returns:
            list: A list of learning rates for each parameter group.
        """
        return [group['lr'] for group in self.param_groups]
