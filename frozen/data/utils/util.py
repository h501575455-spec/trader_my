import time
import pandas as pd
import inspect
import threading
import concurrent.futures
from queue import Queue
from functools import wraps
from typing import TYPE_CHECKING

from ..utils.log import L
logger = L.get_logger("frozen")

if TYPE_CHECKING:
    from ..etl.datafeed import DataFeedManager


# def rate_limiter(max_calls_per_minute):
#     def decorator(func):
#         calls = 0
#         start_time = time.time()

#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             nonlocal calls, start_time
#             calls += 1
#             if calls > max_calls_per_minute:
#                 elapsed_time = time.time() - start_time
#                 if elapsed_time < 10:
#                     sleep_time = 10 - elapsed_time
#                     time.sleep(sleep_time)
#                 calls = 0
#                 start_time = time.time()
#             return func(*args, **kwargs)

#         return wrapper
#     return decorator


def paginated_fetch(max_records_per_call: int = 6000, sleep_between_calls: float = 0.2):
    """
    Pagination decorator for API calls that have record limits.
    
    Parameters
    ----------
    max_records_per_call : int, default 6000
        Maximum records per API call before pagination is triggered
    sleep_between_calls : float, default 0.2
        Sleep time between pagination calls in seconds
        
    Returns
    -------
    decorator
        Function decorator that adds pagination capabilities
        
    Notes
    -----
    This decorator automatically detects when pagination is needed and handles:
    - Record count checking
    - Sleep between calls for rate limiting
    - Configuration through decorator parameters
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject pagination parameters into kwargs if not already provided
            kwargs.setdefault('max_records_per_call', max_records_per_call)
            kwargs.setdefault('sleep_between_calls', sleep_between_calls)
            return func(*args, **kwargs)
        
        # Store pagination config on the function for introspection
        wrapper.pagination_config = {
            'max_records_per_call': max_records_per_call,
            'sleep_between_calls': sleep_between_calls
        }
        return wrapper
    return decorator


def rate_limiter(max_calls_per_minute):
    def decorator(func):
        calls = 0
        start_time = time.time()
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal calls, start_time
            with lock:
                current_time = time.time()
                if current_time - start_time > 60:
                    # Reset the counter and the start time every minute
                    calls = 0
                    start_time = current_time
                if calls >= max_calls_per_minute:
                    # Sleep until the minute is over
                    sleep_time = 60 - (current_time - start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    # Reset after sleeping
                    calls = 0
                    start_time = time.time()
                calls += 1
            return func(*args, **kwargs)

        return wrapper
    return decorator


class verboser(object):

    def __init__(self, verbose=True):
        self.verbose = verbose
 
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(f"Loading data from {args[0]}:")
            result = func(*args, **kwargs)
            if self.verbose:
                print("Done")
            return result
        return wrapper


def worker(queue):
    while True:
        func, args, kwargs = queue.get()
        if func is None:
            break
        func(*args, **kwargs)
        queue.task_done()


def parallel_task(manager: "DataFeedManager", tasks: list):
    """
    Execute multiple data retrieval tasks in parallel
    
    Parameters
    ----------
    manager: DataFeedManager
        The data retrieval manager instance
    tasks: list
        A list of tasks, each task is a tuple containing parameters, the format is as follows:
        - The first element: table name (table_name)
        - The remaining parameters are different depending on the task type
    """
    from ..utils.log import L
    logger = L.get_logger("frozen")
    
    # Create a simple method dictionary
    method_dict = {
        'volume_price': manager.fetch_volume_price_data,
        'stock_limit': manager.fetch_stock_limit_data,
        'stock_fundamental': manager.fetch_stock_fundamental_data,
        'stock_dividend': manager.fetch_stock_dividend_data,
        'stock_suspend': manager.fetch_stock_suspend_data,
        'stock_basic': manager.fetch_stock_basic_data,
        'trade_calendar': manager.fetch_trade_calendar_data,
        'index_weight': manager.fetch_index_weight_data,
        'convertible_bond_basic': manager.fetch_cb_basic_data,
        'convertible_bond_daily': manager.fetch_cb_daily_data,
    }
    
    # Create a table name to method mapping
    table_method_map = {
        'index_daily': 'volume_price',
        'stock_daily_real': 'volume_price',
        'stock_daily_hfq': 'volume_price',
        'stock_daily_limit': 'stock_limit',
        'stock_daily_fundamental': 'stock_fundamental',
        'stock_dividend': 'stock_dividend',
        'stock_suspend_status': 'stock_suspend',
        'stock_basic': 'stock_basic',
        'trade_calendar': 'trade_calendar',
        'index_weight': 'index_weight',
        'convertible_bond_basic': 'convertible_bond_basic',
        'convertible_bond_daily': 'convertible_bond_daily',
    }

    task_queue = Queue()
    num_threads = min(len(tasks), 8)  # Limit the maximum number of threads to 8

    # Define a worker that adjusts parameters intelligently with proper logging
    def worker(queue):
        # Configure logging for this thread - ensure handlers are present
        thread_logger = L.get_logger("frozen")
        # Force re-initialization of logger to ensure handlers are properly set up
        if not thread_logger.handlers:
            L._init_logger()
        
        # Test that logging works in this thread
        thread_logger.info(f"Worker thread {threading.current_thread().name} started")
        
        while True:
            func, args, kwargs = queue.get()
            if func is None:
                thread_logger.info(f"Worker thread {threading.current_thread().name} stopping")
                break
                
            try:
                # Get the function signature
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Check if the number of parameters matches
                if len(args) > len(param_names):
                    # If the number of parameters is too large, only keep the number of parameters needed by the function
                    args = args[:len(param_names)]
                
                # Log the start of task execution
                task_name = args[0] if args else "unknown"
                thread_name = threading.current_thread().name
                thread_logger.info(f"[{thread_name}] Starting task: {task_name}")
                
                func(*args, **kwargs)
                
                # Log the completion of task execution
                thread_logger.info(f"[{thread_name}] Completed task: {task_name}")
                
            except Exception as e:
                thread_logger.error(f"[{threading.current_thread().name}] Error executing task: {e}")
                import traceback
                thread_logger.error(f"[{threading.current_thread().name}] Traceback: {traceback.format_exc()}")
            finally:
                queue.task_done()

    # Log the start of parallel processing
    logger.info(f"Starting parallel processing with {num_threads} threads for {len(tasks)} tasks")

    # Start worker threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(task_queue,), name=f"DataWorker-{i+1}")
        thread.start()
        threads.append(thread)
        logger.info(f"Started thread: {thread.name}")

    # Enqueue tasks
    for i, task in enumerate(tasks):
        table_name = task[0]
        method_key = table_method_map.get(table_name)
        
        if method_key is None:
            logger.warning(f"Unknown table name: {table_name}")
            continue
            
        fetch_func = method_dict.get(method_key)
        
        if fetch_func is None:
            logger.warning(f"No method found for table: {table_name}")
            continue
        
        logger.info(f"Enqueuing task {i+1}/{len(tasks)}: {table_name}")
        # Enqueue the task, the first parameter is the table name, and the remaining parameters are passed to the method by position
        task_queue.put((fetch_func, (table_name,) + task[1:], {}))

    # Wait for all tasks to complete
    logger.info("Waiting for all tasks to complete...")
    task_queue.join()
    logger.info("All tasks completed")

    # Stop worker threads
    logger.info("Stopping worker threads...")
    for i in range(num_threads):
        task_queue.put((None, (), {}))

    for thread in threads:
        thread.join()
        logger.info(f"Thread {thread.name} stopped")
        
    logger.info("Parallel processing completed successfully")


# def parallel_task(manager: "DataFeedManager", tasks: list):

#     from ..etl.datafeed import DataFeedManager
    
#     # fetch_methods = [name for name, obj in DataFeedManager.__dict__.items() if callable(obj) and not name.startswith("_")]
#     methods= inspect.getmembers(DataFeedManager, predicate=inspect.isfunction)
#     fetch_methods = [method for method in methods if not method[0].startswith("_")]

#     task_queue = Queue()
#     num_threads = len(tasks)

#     # Start worker threads
#     threads = []
#     for _ in range(num_threads):
#         thread = threading.Thread(target=worker, args=(task_queue,))
#         thread.start()
#         threads.append(thread)

#     # Enqueue tasks with parameters
#     for param in tasks:
#         # fetch_method = next((method[1] for method in fetch_methods if param[0].split("_")[-2] in method[0].split("_")), None)
#         # fetch_func = fetch_method.__call__
#         # if fetch_func is None:
#         #     # raise ValueError(f"Error, datafeed function for {param} not found!")
#         #     fetch_func = manager.fetch_volumn_price_data
#         fetch_func = next((method[1].__get__(manager, DataFeedManager) for method in fetch_methods if param[0].split("_")[-2] in method[0].split("_")), manager.fetch_volume_price_data)
#         task_queue.put((fetch_func, param, {}))

#     # Block until all tasks are done
#     task_queue.join()

#     # Stop workers
#     for _ in range(num_threads):
#         task_queue.put((None, (), {}))

#     for thread in threads:
#         thread.join()


# def rate_limiter(max_calls_per_minute):
#     lock = threading.Lock()
#     calls = 0
#     start_time = time.time()
    
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             nonlocal calls, start_time
#             with lock:
#                 calls += 1
#                 if calls > max_calls_per_minute:
#                     elapsed_time = time.time() - start_time
#                     if elapsed_time < 60:
#                         sleep_time = 60 - elapsed_time
#                         time.sleep(sleep_time)
#                     calls = 0
#                     start_time = time.time()
#             return func(*args, **kwargs)

#         return wrapper
#     return decorator


# def parallel_task(manager: "DataFeedManager", tasks: list):

#     from ..datafeed import DataFeedManager
    
#     methods= inspect.getmembers(DataFeedManager, predicate=inspect.isfunction)
#     fetch_methods = [method for method in methods if not method[0].startswith("_")]
    
#     num_threads = len(tasks)
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = []
#         for param in tasks:
#             worker = next((method[1].__get__(manager, DataFeedManager) for method in fetch_methods if param[0].split("_")[-2] in method[0].split("_")), manager.fetch_volumn_price_data)
#             futures.append(executor.submit(worker, *param))

#         for future in concurrent.futures.as_completed(futures):
#             future.result()


def convert_datetime_columns(df):
    """
    Recursively convert all date columns to datetime type.
    """
    date_keywords = [
        "date", "trade_date", "ex_date", "list_date", "delist_date", "ann_date", "f_ann_date", "end_date", 
        "cal_date", "pretrade_date", "in_date", "out_date"
        ]
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords) or ("date" in col.split("_")):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception as e:
                logger.error(f"Failed to convert column {col}: {e}")
    
    return df
