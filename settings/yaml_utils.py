from omegaconf import DictConfig, OmegaConf

def insert_value(config: DictConfig, value):
    """
    Recursively replaces any '?' value in a DictConfig object with the given value.
    
    Args:
        config (DictConfig): The configuration dictionary to process.
        value: The value to replace '?' with.
    """
    for key, val in config.items():
        if isinstance(val, DictConfig):
            insert_value(val, value)  # Recursive call for nested dictionaries
        elif val == "?":
            config[key] = value  # Replace '?' with the given value
            
def multiply(first_config: DictConfig, second_config: DictConfig):
    res = []
    for first_config_entry in first_config:
        for second_config_entry in second_config:
            res.append[OmegaConf.merge(first_config_entry,
                                       second_config_entry)]
            
    return DictConfig(res)