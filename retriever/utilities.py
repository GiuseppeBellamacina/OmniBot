import yaml

def load_config(file_path="config.yaml") -> dict:
    """
    Load configuration file
    
    Args:
        file_path (str): Configuration file path
    
    Returns:
        dict: Configuration file
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config