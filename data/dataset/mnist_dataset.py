"""
In this file, we define the Mnist Dataset class.
"""
from omegaconf import DictConfig

from data.dataset.dataset import Dataset, DataType, OutputType
from data.data_sample import DataSample
from settings.serializable import YAMLSerializable
from encoders.encoder import Encoder
from torchvision.datasets import MNIST

@YAMLSerializable.register("MnistDataset")
class MnistDataset(Dataset):
    """
    This class is responsible for loading the MNIST data.
    """
    def __init__(self,
                    root: str,
                    data_type: DataType = DataType.TRAIN,
                    output_type: OutputType = OutputType.TORCH,
                    encoder: Encoder = ...,
                    flatten: bool = False) -> None:

        super().__init__(data_type, output_type, encoder)
        self.root = root
        self.flatten = flatten
        self.data, self.labels = self._load_data()
        
    def _load_data(self):
        """
        Load the MNIST data from the root directory.
        """
        import torchvision.transforms as transforms

        transform = transforms.Compose([transforms.ToTensor()])
        mnist_data = MNIST(self.root, train=self.data_type == DataType.TRAIN, download=True, transform=transform)
        data = mnist_data.data.numpy()
        labels = mnist_data.targets.numpy()
        return data, labels

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data)
    
    def get_raw(self, idx, encoded=True) -> DataSample:
        """
        Returns a single raw item from the dataset
        """
        data = self.data[idx]
        if self.flatten:
            data = data.reshape(-1, 1).tolist()
        else:
            data = data.tolist()
        label = self.labels[idx]
        sample = DataSample(data, label)
        
        if encoded:
            return self._encoder(sample)
        
        return sample

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset
        """
        sample = self.get_raw(idx)
        return Dataset.get(sample, self._output_type), sample.get_label()
    
    @staticmethod
    def from_config(config: DictConfig, 
                    data_type: DataType, 
                    output_type: OutputType, 
                    encoder: Encoder):
        
        return MnistDataset(config.root,
                            flatten=config.flatten,
                            data_type=data_type,
                            output_type=output_type,
                            encoder=encoder)