import torch
import torch.optim as optim

class MomentumOptimizer(optim.Optimizer):
    """
    Implements the SGD optimizer with momentum.
    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        Initializes the optimizer.

        Args:
            params (iterable): An iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.9.
        """
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates   
 the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                if group['momentum'] > 0:
                    param_state = self.state[p]
                    if 'momentum' not in param_state:
                        param_state['momentum'] = torch.clone(d_p).detach()
                    else:
                        param_state['momentum'] = group['momentum'] * param_state['momentum'] + d_p

                    d_p = param_state['momentum']

                p.data.add_(-group['lr'], d_p)

        return loss