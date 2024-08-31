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
            if type(arg) == list:
                print(f'  arg {i+1} of type {type(arg)}: {arg} (length: {len(arg)})')
                for j, item in enumerate(arg[:5]):
                    print(f'    item {j+1} of type {type(item)}: {item}')
            else:
                print(f'  arg {i+1} of type {type(arg)}: {arg}')
        print(f'Function \33[1;36m{func.__name__}\33[0m called with kwargs {kwargs}')
        for key, value in kwargs.items():
            if type(value) == list:
                print(f'  kwarg {key} of type {type(value)}: {value} (length: {len(value)})')
                for j, item in enumerate(value[:5]):
                    print(f'    item {j+1} of type {type(item)}: {item}')
            else:
                print(f'  kwarg {key} of type {type(value)}: {value}')
        try:
            result = func(*args, **kwargs)
            if type(result) == list:
                print(f'Function \33[1;32m{func.__name__}\33[0m returned {result} of type {type(result)} (length: {len(result)})')
                for i, item in enumerate(result[:5]):
                    print(f'  item {i+1} of type {type(item)}: {item}')
            else:
                print(f'Function \33[1;32m{func.__name__}\33[0m returned {result} of type {type(result)}')
            print("\33[1;33m----------------------------------------------\33[0m")
            return result
        except Exception as e:
            print(f'Function \33[1;31m{func.__name__}\33[0m raised an exception: {e}')
            print("\33[1;33m----------------------------------------------\33[0m")
            raise e
    return wrapper