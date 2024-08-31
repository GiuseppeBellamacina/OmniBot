import yaml
import inspect

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
def debug(max_items=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\33[1;33m----------------------------------------------\33[0m")
            print("\33[1;33m[DEBUGGER]\33[0m")
            print(f"\33[1;37mNAME\33[0m: \33[1;36m{func.__name__}\33[0m")
            
            sig = inspect.signature(func)
            params = sig.parameters
            
            print(f'\33[1;37mARGS\33[0m:')
            for _, (name, arg) in enumerate(zip(params, args)):
                print_arg_info(name, arg, max_items)
            
            if kwargs:
                print(f'\33[1;37mKWARGS\33[0m:')
                for key, value in kwargs.items():
                    print_arg_info(key, value, max_items)
            
            try:
                result = func(*args, **kwargs)
                print("\33[1;37mSTATUS\33[0m: \33[1;32mOK\33[0m")
                print(f'\33[1;37mRETURN\33[0m:')
                print_return_info(result, max_items)
                print("\33[1;33m----------------------------------------------\33[0m")
                return result
            except Exception as e:
                print("\33[1;37mSTATUS\33[0m: \33[1;31mERROR\33[0m")
                print(f'Function \33[1;31m{func.__name__}\33[0m raised an exception: {e}')
                print("\33[1;33m----------------------------------------------\33[0m")
                raise e
        return wrapper
    return decorator

def print_arg_info(param_name, arg, max_items):
    if isinstance(arg, (list, tuple, set)):
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}) with {len(arg)} items:')
        for j, item in enumerate(list(arg)[:max_items]):
            print(f'    item {j+1} of type {type(item)}: {item}')
    elif isinstance(arg, dict):
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}) with {len(arg)} items:')
        for j, (key, value) in enumerate(list(arg.items())[:max_items]):
            print(f'    key {j+1} of type {type(key)}: {key}, value of type {type(value)}: {value}')
    else:
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}): {arg}')

def print_return_info(result, max_items):
    if isinstance(result, (list, tuple, set)):
        print(f'  Function returned {type(result)} with {len(result)} items:')
        for i, item in enumerate(list(result)[:max_items]):
            print(f'    item {i+1} of type {type(item)}: {item}')
    elif isinstance(result, dict):
        print(f'  Function returned {type(result)} with {len(result)} items:')
        for i, (key, value) in enumerate(list(result.items())[:max_items]):
            print(f'    key {i+1} of type {type(key)}: {key}, value of type {type(value)}: {value}')
    else:
        print(f'  Function returned {type(result)}: {result}')