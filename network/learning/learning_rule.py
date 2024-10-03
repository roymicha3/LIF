import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from network.connection import Connection


class LearningRule(torch.autograd.Function):
    """
    Base class for learning rules )(simple learning rule for fully connected linear layer).
    """

    def __init__(
        self,
        connection: Connection,
        lr,
        **kwargs,
    ) -> None:
        """
        Constructor for the ``LearningRule`` object.
        :param connection: An ``Connection`` object.
        :param lr: learning rates for pre- and post-synaptic events
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        self.lr = lr

    def update(self, **kwargs) -> None:
        """
        learning rule update.
        """
        # Bound weights.
        if ((self.connection.wmin != -np.inf).any() or (self.connection.wmax != np.inf).any()):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)

    @staticmethod
    def forward(ctx, input_, weight_) -> any:
        """
        forward function for the learning rule
        """
        # Save input_ and weight_ for backward computation
        ctx.save_for_backward(input_, weight_)
        # Perform forward computation (simple linear transformation)
        output = input_.mm(weight_.t())
        return output

    @staticmethod
    def backward(ctx: any, grad_output: any) -> tuple:
        """
        backward function for the learning rule
        """
        # Retrieve saved tensors
        input_, weight = ctx.saved_tensors
        # Compute gradients w.r.t. inputs and weights
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input_)
        return grad_input, grad_weight
