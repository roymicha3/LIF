"""
This module contains the Configuration class, which is used to store the attributes of a model
"""
import json

class Configuration:
    """
    A class to store the attributes of a model
    """

    def __init__(self, attr: dict):
        self._attributes = attr

    def __getitem__(self, key):
        return self._attributes[key]

    def __setitem__(self, key, value):
        self._attributes[key] = value

    def __len__(self):
        return len(self._attributes)

    def dump(self, filename):
        """
        Dump the model attributes to a JSON file.
        """
        attributes = {attr: getattr(self, attr) for attr in self._attributes}
        with open(filename, "wb") as f:
            json.dump(attributes, f, indent=4)

    def load(self, filename):
        """
        Load model attributes from a JSON file.
        """
        with open(filename, "rb") as f:
            attributes = json.load(f)

        for attr, value in attributes.items():
            self[attr] = value

    def summarize(self):
        """
        Print a summary of the model attributes, including their names and values.
        """
        print("Model Attribute Summary:")
        for attr in self._attributes:
            value = getattr(self, attr)
            print(f"\t- {attr}: {value}")
