"""
This module contains the ModelAttributes class, which is used to store the attributes of a model
"""

class ModelAttributes:
    """
    A class to store the attributes of a model
    """
    def __init__(self, **kwargs):
        self.__attributes = set(kwargs.keys())

        for attr in self.__attributes:
            value = kwargs.get(attr, None)
            if value is None:
                print(f"Warning: {attr} not found in kwargs. Defaulting to None.")
            setattr(self, attr, value)

    def get_attr(self, attr, default=None):
        """
        Get an attribute value. If the attribute doesn't exist, return the default value.
        """
        if attr not in self.__attributes:
            print(f"Warning: {attr} not found. Returning default value.")
        return getattr(self, attr, default)

    def add_attr(self, attr, value=None):
        """
        Add a new attribute to the model. If no value is provided, default to None.
        """
        self.__attributes.add(attr)
        setattr(self, attr, value)
