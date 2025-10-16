import pandas as pd
from functools import wraps
from .helper import _lazy_import_factor
from .anime import Spinner


class Decorator:
    '''Decorator that prints terminal messages.'''
    
    @staticmethod
    def factor(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print('Calculating factor:')
            result = func(*args, **kwargs)
            if isinstance(result, _lazy_import_factor()):
                print(result.expr)
            elif isinstance(result, pd.DataFrame):
                print(result.columns[0])
            else:
                pass
            return result
        return wrapper
    

    @staticmethod
    def result(attribute_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                print_enabled = getattr(self.__class__, attribute_name, True)
                result = func(self, *args, **kwargs)
                if print_enabled:
                    print('🎉 Execution complete!')
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def framework(attribute_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                print_enabled = getattr(self.__class__, attribute_name, True)
                if print_enabled:
                    print("🧊 Initializing FROZEN Backtest Framework...")
                    # time.sleep(1)
                result = func(self, *args, **kwargs)
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def marketdata(attribute_name):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                print_enabled = getattr(self.__class__, attribute_name, True)
                if print_enabled:
                    spinner = Spinner()
                    spinner.start(message="Connecting to database... (Press Ctrl+C to interrupt)")
                success = True
                try:
                    result = func(self, *args, **kwargs)
                except:
                    success = False
                if print_enabled:
                    spinner.stop(success)
                return result
            return wrapper
        return decorator

decorator = Decorator()
