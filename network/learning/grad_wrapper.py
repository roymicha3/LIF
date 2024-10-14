import torch

class GradWrapper:
    """
    A wrapper class for gradient tensors, providing additional information and methods for manipulation.

    Args:
        output_grad (torch.Tensor): The underlying gradient tensor.
        info (dict, optional): A dictionary containing additional information about the gradient.
    """

    def __init__(self, output_grad: torch.Tensor, info: dict = {}):
        self._output_grad = output_grad
        self._info = info

    @property
    def output_grad(self) -> torch.Tensor:
        """
        Gets the underlying gradient tensor.
        """
        return self._output_grad

    @output_grad.setter
    def output_grad(self, new_output_grad: torch.Tensor):
        """
        Sets the underlying gradient tensor.
        """
        self._output_grad = new_output_grad

    @property
    def info(self) -> dict:
        """
        Gets the additional information dictionary.
        """
        return self._info

    @info.setter
    def info(self, new_info: dict):
        """
        Sets the additional information dictionary.
        """
        self._info = new_info

    def additional_info(self, key) -> any:
        """
        Gets the value associated with a specific key in the additional information dictionary.

        Args:
            key (str): The key to look up.

        Returns:
            The value associated with the key, or None if the key is not found.
        """
        return self._info.get(key)


class ConnectionGradWrapper(GradWrapper):
    """
    A specialized GradWrapper for connection gradients, providing additional methods for accessing the weight gradient.

    Args:
        output_grad (torch.Tensor): The gradient with respect to the output.
        weight_grad (torch.Tensor): The gradient with respect to the weights.
        info (dict, optional): A dictionary containing additional information about the connection gradient.
    """

    def __init__(self, output_grad: torch.Tensor, weight_grad: torch.Tensor, info: dict = {}):
        super().__init__(output_grad, info)
        self._weight_grad = weight_grad

    @property
    def weight_grad(self) -> torch.Tensor:
        """
        Gets the gradient with respect to the weights.
        """
        return self._weight_grad
    
    @weight_grad.setter
    def weight_grad(self, new_weight_grad):
        """
        Sets the gradient with respect to the weights.
        """
        self._weight_grad = new_weight_grad