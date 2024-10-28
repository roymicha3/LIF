import torch

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
