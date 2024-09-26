from inspect import signature

def debug(max_items=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\33[1;33m----------------------------------------------\33[0m")
            print("\33[1;33m[DEBUGGER]\33[0m")
            print(f"\33[1;37mNAME\33[0m: \33[1;36m{func.__name__}\33[0m")
            
            sig = signature(func)
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

def print_dict(obj, max_items, indent=4):
    for i, (key, value) in enumerate(list(obj.items())[:max_items]):
        if isinstance(value, (list, tuple, set)):
            print(f'{" "*indent}key {i+1} of type {type(key)}: {key}, value of type {type(value)} with {len(value)} items:')
            print_subscritable(value, max_items, indent+2)
        elif isinstance(value, dict):
            print(f'{" "*indent}key {i+1} of type {type(key)}: {key}, value of type {type(value)} with {len(value)} items:')
            print_dict(value, max_items, indent+2)
        else:
            if len(str(value)) > 50:
                print(f'{" "*indent}key {i+1} of type {type(key)}: {key}, value of type {type(value)}: {str(value)[:50]}...')
            else:
                print(f'{" "*indent}key {i+1} of type {type(key)}: {key}, value of type {type(value)}: {value}')
    if len(obj) > max_items:
        print(f'{" "*indent}...')

def print_subscritable(obj, max_items, indent=4):
    for i, item in enumerate(list(obj)[:max_items]):
        if isinstance(item, (list, tuple, set)):
            print(f'{" "*indent}item {i+1} of type {type(item)} with {len(item)} items:')
            print_subscritable(item, max_items, indent+2)
        elif isinstance(item, dict):
            print(f'{" "*indent}item {i+1} of type {type(item)} with {len(item)} items:')
            print_dict(item, max_items, indent+2)
        else:
            if len(str(item)) > 50:
                print(f'{" "*indent}item {i+1} of type {type(item)}: {str(item)[:50]}...')
            else:
                print(f'{" "*indent}item {i+1} of type {type(item)}: {item}')
    if len(obj) > max_items:
        print(f'{" "*indent}...')

def print_arg_info(param_name, arg, max_items):
    if isinstance(arg, (list, tuple, set)):
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}) with {len(arg)} items:')
        print_subscritable(arg, max_items)
    elif isinstance(arg, dict):
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}) with {len(arg)} items:')
        print_dict(arg, max_items)
    else:
        print(f'  \33[1;35m{param_name}\33[0m (type: {type(arg)}): {arg}')

def print_return_info(result, max_items):
    if isinstance(result, (list, tuple, set)):
        print(f'  Function returned {type(result)} with {len(result)} items:')
        print_subscritable(result, max_items)
    elif isinstance(result, dict):
        print(f'  Function returned {type(result)} with {len(result)} items:')
        print_dict(result, max_items)
    else:
        print(f'  Function returned {type(result)}: {result}')