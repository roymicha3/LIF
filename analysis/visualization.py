"""
Here we visualize our data and results
"""
from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from tools.utils import SEQ_LEN
from common import ATTR, SPIKE_NS, MODEL_NS

DEFAULT_BATCH = 0

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        batch_size = batch_size if batch_size is not DEFAULT_BATCH else ATTR(MODEL_NS.BATCH_SIZE)
        
        T = ATTR(SPIKE_NS.T)
        dt = ATTR(SPIKE_NS.dt)
        self._dataset = RandomDataset(
            ATTR(MODEL_NS.NUM_INPUTS), 
            SEQ_LEN(T, dt),
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(1))
    
    
    def single_spike_raster(self):
        """
        plots rasters of the random spike patterns
        """
        
        FIRST_IDX = 0
        data = self._dataset.get_raw(FIRST_IDX)
        
        data.plot()
