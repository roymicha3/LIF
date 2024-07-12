"""
Here we visualize our data and results
"""
from encoders.spike.single_spike_encoder import SingleSpikeEncoder
from data.loaders.random_data_loader import RandomDataLoader
from common import ATTR, SPIKE_NS, MODEL_NS

DEFAULT_BATCH = 0

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        batch_size = batch_size if batch_size is not DEFAULT_BATCH else ATTR(MODEL_NS.BATCH_SIZE)
        
        self.__loader = RandomDataLoader(
            batch_size,
            SingleSpikeEncoder(ATTR(SPIKE_NS.T)),
            ATTR(SPIKE_NS.T))
    
    
    def single_spike_raster(self):
        """
        plots rasters of the random spike patterns
        """
        batch = self.__loader.load_batch()
        
        FIRST_IDX = 0
        data = batch[FIRST_IDX]
        
        data.plot()
