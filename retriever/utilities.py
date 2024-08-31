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

# decorator for debugging
def debug(func):
    def wrapper(*args, **kwargs):
        print("\33[1;33m----------------------------------------------\33[0m")
        print("\33[1;33m[DEBUGGER]\33[0m")
        print(f'Function \33[1;36m{func.__name__}\33[0m called with args:')
        for i, arg in enumerate(args):
            print(f'  arg {i+1} of type {type(arg)}: {arg}')
        print(f'Function \33[1;36m{func.__name__}\33[0m called with kwargs {kwargs}')
        for key, value in kwargs.items():
            print(f'  kwarg {key} of type {type(value)}: {value}')
        try:
            result = func(*args, **kwargs)
            print(f'Function \33[1;32m{func.__name__}\33[0m returned {result}')
            print("\33[1;33m----------------------------------------------\33[0m")
            return result
        except Exception as e:
            print(f'Function \33[1;31m{func.__name__}\33[0m raised an exception: {e}')
            print("\33[1;33m----------------------------------------------\33[0m")
            raise e
    return wrapper