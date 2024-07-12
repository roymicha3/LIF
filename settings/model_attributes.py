"""
This module contains the ModelAttributes class, which is used to store the attributes of a model
"""
import json

class ModelAttributes:
    """
    A class to store the attributes of a model
    """

    INSTANCE = None

    def __init__(self, **kwargs):
        self.__attributes = set(kwargs.keys())
        ModelAttributes.INSTANCE = self

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

    @staticmethod
    def get_model_attributes():
        """
        Get the model attributes instance.
        """
        if ModelAttributes.INSTANCE is None:
            print("Warning: ModelAttributes instance not found. Creating a new one.")
            return ModelAttributes()

        return ModelAttributes.INSTANCE

    def dump(self, filename):
        """
        Dump the model attributes to a JSON file.
        """
        attributes = {attr: getattr(self, attr) for attr in self.__attributes}
        with open(filename, "w") as f:
            json.dump(attributes, f, indent=4)

    def load(self, filename):
        """
        Load model attributes from a JSON file.
        """
        with open(filename, "r") as f:
            attributes = json.load(f)

        for attr, value in attributes.items():
            self.add_attr(attr, value)

    def summarize(self):
        """
        Print a summary of the model attributes, including their names and values.
        """
        print("Model Attribute Summary:")
        for attr in self.__attributes:
            value = getattr(self, attr)
            print(f"\t- {attr}: {value}")
