import torch

class LR:
    
    def __init__(self):
        self.saved_tensors = None
        self.max_values = None
    
    
    def forward(self, input_, output_) -> torch.Tensor:
    
        if self.max_values is None:
            self.max_values = output_.clone()
        
            self.saved_tensors = torch.zeros(size=(input_.size(0),
                                                input_.size(1),
                                                output_.size(-1)),
                                            dtype=torch.float32,
                                            device=input_.device)
            for n in range(output_.size(-1)):
                self.saved_tensors[:, :, n] = input_.clone()
            
            
        indices = (self.max_values < output_)
        
        for n in range(output_.size(-1)):
            self.saved_tensors[indices[:, n], :, n] = input_[indices[:, n]]
        
        self.max_values[indices] = output_[indices]
        
        return self.max_values
    
    def reset(self):
        self.saved_tensors = None
        self.max_values = None
    
B = 64
T = 500
N = 1000
d = 1

epochs = 100

lr = LR()

for epoch in range(epochs):
    x = torch.randn(B, N, T)
    W = torch.randn(N, d)

    y = x.permute(0, 2, 1) @ W
    y = y.permute(0, 2, 1)


    for i in range(T):
        res = lr.forward(x[:, :, i], y[:, :, i])

    indices = torch.argmax(y, dim=-1)

    for b in range(B):
        
        for n in range(N):
            org_val = x[b, n, indices[b]]
            val = lr.saved_tensors[b, n]
            
            if not torch.isclose(org_val, val):
                print(f"Caught an error at epoch {epoch} for batch {b} and neuron {n}")
                print(f"Original value {org_val} does not match LR value {val}")
                exit(1)
        
        for dim in range(d): 
            if not torch.isclose(y.max(dim=-1)[0][b, dim], res[b, dim]):
                print(f"Caught an error at epoch {epoch} for batch {b}")
                print(f"Original value {y.max(dim=-1)[0][b, dim]} does not match LR value {val[b, dim]}")
                exit(1)
                
    lr.reset()
    print(f"Epoch {epoch} finished without errors")
            
print("All values match!")