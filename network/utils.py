import torch
import torch.nn.functional as F

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='checkpoint.pth', save=False):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Print message when stopping early.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save = save

    def __call__(self, val_loss, model):
        """
        Check if validation loss has improved; otherwise increase counter.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save if it achieves a new best validation loss.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save:
                torch.save(model.state_dict(), self.path)  # Save the model as a checkpoint
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

import torch
import torch.nn.functional as F

def find_spike(tensor, threshold):
    # Ensure tensor has at least 3 elements to find a local minimum
    if tensor.numel() < 3:
        return -1

    # Add singleton dimension and reshape for convolution
    tensor = tensor.view(1, 1, -1)

    # Create a kernel to detect local minima by checking neighbors
    kernel = torch.tensor([1.0, -2.0, 1.0], device=tensor.device).view(1, 1, -1)
    
    # Apply 1D convolution with padding to keep dimensions the same
    convolved = F.conv1d(tensor, kernel, padding=1)

    # Find indices of local minima
    local_minima_mask = (convolved < 0).squeeze()

    # Filter local minima that are above the threshold
    tensor_flat = tensor.squeeze()
    valid_minima_mask = local_minima_mask & (tensor_flat > threshold)
    valid_indices = torch.nonzero(valid_minima_mask).squeeze()

    # Return the index of the first valid local minimum, or None if none found
    if valid_indices.numel() <= 0:
        return -1
    
    if valid_indices.numel() == 1:
        return valid_indices.item()
    
    return valid_indices[0].item()

