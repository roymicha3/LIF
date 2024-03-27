"""
a module that provides common functions for the model.
"""
from settings.model_attributes import ModelAttributes


def get_model_attributes():
    """
    Get the model attributes instance.
    """
    return ModelAttributes.get_model_attributes()

ATTR = get_model_attributes
