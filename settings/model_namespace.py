"""
in this file we define the attributes of the model
"""

class ModelNamespace(object):
    """
    A class to store the attributes of a spiking model
    """
    # model attributes
    NUM_INPUTS = "num_inputs"
    NUM_OUTPUTS = "num_outputs"
    
    # data attributes
    NUM_CLASSES = "num_classes"
    BATCH_SIZE = "batch_size"
    TRAINING_PERCENTAGE = "training_percentage"
    VALIDATION_PERCENTAGE = "validation_percentage"
    TESTING_PERCENTAGE = "testing_percentage"
