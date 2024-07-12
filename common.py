"""
a module that provides common functions for the model.
"""
from settings.model_attributes import ModelAttributes
from settings.spike.spike_namespace import SpikeNamespace
from settings.model_namespace import ModelNamespace


def get_model_attributes():
    """
    Get the model attributes instance.
    """
    return ModelAttributes.get_model_attributes()


def init_model_attributes(values: dict):
    """
    Initializes the attributes for the model
    """
    for key, val in values.items():
        get_model_attributes().add_attr(key, val)
        
    get_model_attributes().summarize()


ATTR = get_model_attributes().get_attr
SPIKE_NS = SpikeNamespace
MODEL_NS = ModelNamespace
