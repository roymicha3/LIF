from data_loader import DataLoader, DataType

class MnistDataLoader(DataLoader):
    
    def __init__(self, batch_size, encoder):
        super().__init__(batch_size, encoder)
        
    def load(self, type: DataType):
        pass
    
    def load_batch(self, type: DataType):
        pass
    
    def __call__(self, type: DataType):
        pass