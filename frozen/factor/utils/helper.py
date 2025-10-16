import time
from functools import wraps
from typing import Callable, Any

def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A timer decorator that Records the running time 
    of a function and prints related information

    Features:
        - Records the start and end time of the function 
          and calculates the elapsed time
        - Prints the function name, input parameters, 
          return value (optional), and running time
        - Supports nested functions and functions with 
          parameters

    Parameter:
        func: The function to be decorated

    Returns:
        The wrapped function that retains the original 
        function's functionality with additional timing logic
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            func_name = func.__name__
            # args_repr = [repr(a) for a in args]
            # kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            # signature = ", ".join(args_repr + kwargs_repr)
            
            print(f"Func {func_name} call completed")
            print(f"TIme spent: {elapsed_time:.6f}s")
    
    return wrapper


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value