"""
in this file we define the attributes of the spike model
"""
from settings.model_attributes import ModelAttributes

class SpikeAttributes(ModelAttributes):
    """
    A class to store the attributes of a model
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_T(self):
        """
        returns the time of the simulation (in ms)
        """
        return self.get_attr('T')

    def get_dt(self):
        """
        returns the time step of the simulation (in ms)
        """
        return self.get_attr('dt')

    def get_num_of_neurons(self):
        """
        returns the number of neurons in the model
        """
        return self.get_attr('num_of_neurons')
    