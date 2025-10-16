import ast
import sys
import inspect
import numpy as np
import pandas as pd
from functools import wraps
from collections import defaultdict
from joblib import Parallel, delayed

from ..base import Factor
from .formatter import str2expr, _get_ops_names
from ...utils.preprocess import normalize, standardize, clip_mad, winsorize_quantile


##############################################################
# Decorator for preprocess operators

def prep(
        method="normalize",
        cross_section=False,
        expanding=False,
        window=None,
        limits=(0.1, 0.1),
    ):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = func(*args, **kwargs)
            if method == "normalize":
                res = normalize(data, cross_section, expanding=expanding, window=window)
            elif method == "standardize":
                res = standardize(data, cross_section, expanding=expanding, window=window)
            elif method == "clip":
                res = clip_mad(data, expanding, window)
            elif method == "winsorize":
                res = winsorize_quantile(data, limits, expanding, window)
            else:
                raise NotImplementedError("Preprocessing operator not implemented yet!")
            return res
        return wrapper
    return decorator

##############################################################
# Calculate factor expression (by batch)

def calc_str(
        expr_str: str,
        additional_vars: dict = None,
        piecewise: bool = False
    ) -> Factor:
    """
    Calculate single factor.
    """
    from .. import operators
    from ...utils import preprocess
    try:
        _ops_names = _get_ops_names()
        ops_attr = vars(operators)  # obtain all attributes from operations module
        prep_attr = vars(preprocess)
        attr = {**ops_attr, **prep_attr}
        namespace = dict(filter(lambda item: item[0] in _ops_names, attr.items()))
        if additional_vars:
            namespace.update(additional_vars)
        result = eval(str2expr(expr_str), namespace)
        if piecewise:
            data = pd.DataFrame(result.data.iloc[-1]).T
            return Factor(data, expr_str)
        else:
            return result
    except NameError as e:
        print(f"NameError: {e}")
        print("Available names in namespace:", list(namespace.keys()))
        raise


def batch_calc(
        expr_list: list,
        name_list: list,
        additional_vars: dict = None,
        piecewise: bool = False,
        parallel: bool = False
    ) -> pd.DataFrame:

    factor = dict()

    if parallel:
        results = Parallel(n_jobs=-1, verbose=1, prefer="threads")(
            delayed(calc_str)(expr, additional_vars, piecewise) for expr in expr_list)
        for name, alpha in zip(name_list, results):
            factor[name] = alpha.data.stack().swaplevel().sort_index()

    else:
        for expr, name in zip(expr_list, name_list):
            alpha = calc_str(expr, additional_vars, piecewise)
            factor[name] = alpha.data.stack().swaplevel().sort_index()
    
    if len(factor) > 1:
        return pd.concat(factor, axis=1)
    else:
        return pd.DataFrame(factor)

##############################################################
# Parse expression and get safe lookback period

def check_parameter(func, param_name):
    sig = inspect.signature(func)
    return param_name in sig.parameters


def get_parameter_position(func, param_name):
    """Obtain the position of the parameter in the function (starting from 0)"""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    return params.index(param_name) if param_name in params else None


class ExpressionAnalyzer(ast.NodeVisitor):
    def __init__(self, target_param="window"):
        self.target_param = target_param
        self.param_values = {}  # the explicit window value of the function
        self.sliding_chains = []  # total value of the sliding window chain
        self.current_chain = []  # current sliding window function chain
        self.module = sys.modules["frozen.factor.expression.operators"]
        self.func_dependencies = defaultdict(list)  # function dependencies
        self.call_counter = defaultdict(int)  # function call counter
        self.call_details = []  # all function call details
        self.func_signatures = {}  # function signatures
    
    def visit_Call(self, node):
        # current function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            call_id = f"{func_name}_{self.call_counter[func_name]}"
            self.call_counter[func_name] += 1
            
            # current call details
            call_info = {
                "id": call_id,
                "name": func_name,
                "args": [],
                "window_value": None,
                "has_window_param": False,
                "is_sliding": False
            }
            self.call_details.append(call_info)
            
            # record function dependencies
            for arg in ast.walk(node):
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg != node:
                    # record all sub-calls
                    self.func_dependencies[call_id].append(arg.func.id)
            
            try:
                # the actual function object
                func = getattr(self.module, func_name)
                
                # function signature
                sig = inspect.signature(func)
                self.func_signatures[func_name] = {
                    "parameters": list(sig.parameters.keys()),
                    "has_window": self.target_param in sig.parameters
                }
                
                # check if the function has the target parameter
                if self.func_signatures[func_name]["has_window"]:
                    call_info["has_window_param"] = True
                    
                    # find the specified parameter value
                    window_value = None
                    
                    # 1. check by position
                    target_pos = get_parameter_position(func, self.target_param)
                    if target_pos is not None and target_pos < len(node.args):
                        arg = node.args[target_pos]
                        if isinstance(arg, ast.Constant):
                            window_value = arg.value
                    
                    # 2. check by keyword
                    for kw in node.keywords:
                        if kw.arg == self.target_param:
                            if isinstance(kw.value, ast.Constant):
                                window_value = kw.value.value
                            break
                    
                    # record window value
                    if window_value is not None:
                        # store the value with call ID to avoid overwrite
                        self.param_values[call_id] = window_value
                        call_info["window_value"] = window_value
                        
                        # if the function has a sliding window
                        if check_parameter(func, self.target_param):
                            call_info["is_sliding"] = True
                            # add to the current chain
                            self.current_chain.append((call_id, window_value))
            
            except AttributeError:
                # ignore the undefined function
                pass
        
        # recursive processing of sub-nodes
        self.generic_visit(node)
        
        # after the current function is processed, check the sliding window chain
        if call_info.get("is_sliding", False):
            # if the current function has a sliding window and the chain has multiple functions
            if len(self.current_chain) > 1:
                # calculate the total value of the current sliding window chain
                chain_values = [w for _, w in self.current_chain]
                total = sum(chain_values) - (len(chain_values) - 1)
                self.sliding_chains.append(total)
            
            # remove current function from chain
            if self.current_chain and self.current_chain[-1][0] == call_id:
                self.current_chain.pop()

def analyze_expression(expr, target_param="window"):
    """Analyze the parameter values in the expression"""    
    try:
        tree = ast.parse(expr)
        analyzer = ExpressionAnalyzer(target_param)
        analyzer.visit(tree)
        
        # convert the call details to a more readable format
        param_values = {}
        for call in analyzer.call_details:
            if call["window_value"] is not None:
                # use function name + index as key
                key = f"{call['name']}_{call['id'].split('_')[-1]}"
                param_values[key] = call["window_value"]
        
        # convert function dependencies
        func_dependencies = defaultdict(list)
        for caller, callees in analyzer.func_dependencies.items():
            caller_name = caller.split("_")[0]
            func_dependencies[caller_name].extend(callees)
        
        # collect all function signatures
        func_signatures = {}
        for func_name, sig_info in analyzer.func_signatures.items():
            func_signatures[func_name] = {
                "parameters": sig_info["parameters"],
                "has_window": sig_info["has_window"]
            }
        
        return {
            "param_values": param_values,
            "sliding_chains": analyzer.sliding_chains,
            "func_dependencies": dict(func_dependencies),
            "call_details": analyzer.call_details,
            "func_signatures": func_signatures
        }
    except SyntaxError:
        return {
            "param_values": {},
            "sliding_chains": [],
            "func_dependencies": {},
            "call_details": [],
            "func_signatures": {}
        }

def get_safe_lookback_period(expr_str, print_detail=False):
    """
    Calculate the maximum lookback window needed for the factor
    so no NaN values should occur.

    - nested sliding window func: sum the windows and minus (n-1)
        e.g. window_1 + window_2 - 1
    - parallel func: take the maximum
    """
    expr = str2expr(expr_str)
    
    analysis = analyze_expression(expr, "window")
    param_values = analysis["param_values"]
    sliding_chains = analysis["sliding_chains"]
    func_dependencies = analysis["func_dependencies"]
    call_details = analysis["call_details"]
    func_signatures = analysis["func_signatures"]
    
    all_values = []
    
    # 1. Add sliding window chain (for nested sliding window)
    all_values.extend(sliding_chains)
    
    # 2. Add explicit sliding window value (for single sliding window)
    for call in call_details:
        if call["window_value"] is not None and call["is_sliding"]:
            # Check if processed in chain
            in_chain = any(call["id"] in chain_info for chain_info in func_dependencies.values())
            if not in_chain:
                all_values.append(call["window_value"])
    
    # # 3. Add non sliding window function but with window parameter
    # for call in call_details:
    #     if call["has_window_param"] and not call["is_sliding"] and call["window_value"] is not None:
    #         all_values.append(call["window_value"])
    
    if print_detail:
        print("Function signature:")
        for func, sig in func_signatures.items():
            window_status = "with `window` parameter" if sig["has_window"] else "no `window` parameter"
            print(f"  {func}(): {window_status}, parameters: {', '.join(sig['parameters'])}")
        
        print("Function call detail:")
        for call in call_details:
            details = f"{call['id']}: {call['name']}"
            
            if call["has_window_param"]:
                details += " [with `window` parameter]"
                if call["window_value"] is not None:
                    details += f", window={call['window_value']}"
                if call["is_sliding"]:
                    details += " [sliding window function]"
            
            print(f"  {details}")
        
        print("Function dependency:")
        for caller, callees in func_dependencies.items():
            print(f"  {caller}() called: {', '.join(callees)}")
        
        if param_values:
            print("Explicit window values:")
            for key, value in param_values.items():
                print(f"  {key}: {value}")
        
        if sliding_chains:
            print("Sliding window chain total:")
            for i, total in enumerate(sliding_chains, 1):
                print(f"  chain{i}: {total}")
        
        print(f"All calculated values: {all_values}")
    
    # Calculate max lookback period
    if all_values:
        max_value = max(all_values)
        return min(max_value, 252)
    else:
        # no sliding window, set default as 0
        return 0


##############################################################
# Optimized way to build pandas data structure

class OptimizedFactor:
    """
    Optimized Factor class with lazy Pandas reconstruction.
    
    This class delays expensive Pandas operations until actually needed,
    providing significant performance improvements for large datasets.
    """
    
    def __init__(self, raw_data, expr, param_type, time_index, ticker_columns, 
                 window, n_windows, lazy_rebuild=True, is_multi_factor=False, factor_exprs=None):
        self.raw_data = raw_data
        self.expr = expr
        self.param_type = param_type
        self.time_index = time_index
        self.ticker_columns = ticker_columns
        self.window = window
        self.n_windows = n_windows
        self.is_multi_factor = is_multi_factor
        self.factor_exprs = factor_exprs or []  # List of factor expressions for multi-factor
        self._pandas_cache = None
        self._partial_cache = {}  # Cache for partial data access
        
        # If lazy rebuild is disabled, build immediately
        if not lazy_rebuild:
            self._pandas_cache = self._build_pandas_structure()
    
    @property
    def data(self):
        """Lazy evaluation of Pandas structure."""
        if self._pandas_cache is None:
            # print(f"🔄 Building Pandas structure for {self.param_type}...")
            self._pandas_cache = self._build_pandas_structure()
        return self._pandas_cache
    
    def get_ticker_data(self, ticker):
        """Get data for a specific ticker without building full structure."""
        if ticker not in self.ticker_columns:
            raise ValueError(f"Ticker {ticker} not found")
        
        if ticker in self._partial_cache:
            return self._partial_cache[ticker]
        
        ticker_idx = list(self.ticker_columns).index(ticker)
        
        if self.param_type == "resid":
            # Extract residuals for specific ticker
            ticker_data = self.raw_data[:, :, ticker_idx]  # Shape: (n_windows, window)
            
            # Create index for this ticker
            window_dates = self.time_index[self.window-1:self.window-1+self.n_windows]
            
            result_list = []
            for w_idx, date in enumerate(window_dates):
                for window_pos in range(self.window):
                    result_list.append({
                        'trade_date': date,
                        'ticker': ticker,
                        'window': window_pos,
                        'resid': ticker_data[w_idx, window_pos]
                    })
            
            ticker_df = pd.DataFrame(result_list)
            ticker_result = ticker_df.set_index(['trade_date', 'ticker', 'window'])['resid']
            
        else:  # beta or alpha
            # Extract coefficients for specific ticker
            ticker_coef = self.raw_data[:, 0, ticker_idx]  # Shape: (n_windows,)
            
            # Create 2D result
            full_result = np.full(len(self.time_index), np.nan)
            window_dates = self.time_index[self.window-1:self.window-1+self.n_windows]
            
            for i, date in enumerate(window_dates):
                date_idx = self.time_index.get_loc(date)
                full_result[date_idx] = ticker_coef[i]
            
            ticker_result = pd.Series(full_result, index=self.time_index, name=ticker)
        
        self._partial_cache[ticker] = ticker_result
        return ticker_result
    
    def _build_pandas_structure(self):
        """Build the appropriate Pandas structure based on parameter type."""
        if self.param_type == "resid":
            return self._build_resid_structure()
        elif self.param_type == "beta" and self.is_multi_factor:
            return self._build_multi_factor_structure()
        else:
            return self._build_2d_structure()
    
    def _build_resid_structure(self):
        """Build residual structure with 2-level MultiIndex and window as columns - Optimized with full index alignment."""
        n_windows, window, n_tickers = self.raw_data.shape
        total_elements = len(self.time_index) * window * n_tickers
        
        # print(f"🔄 Building residual structure: {len(self.time_index)}×{window}×{n_tickers} = {total_elements:,} elements")
        
        # PERFORMANCE OPTIMIZED: Use vectorized operations with full index alignment
        
        # Step 1: Create full-size result array with NaN
        full_data = np.full((len(self.time_index), n_tickers, window), np.nan, dtype=np.float64)
        
        # Step 2: Fill in valid data using vectorized assignment
        # Map window_dates to full time index positions
        window_dates = self.time_index[self.window-1:self.window-1+n_windows]
        valid_positions = np.searchsorted(self.time_index, window_dates)
        
        # Vectorized assignment: full_data[valid_positions, :, :] = raw_data
        # Need to transpose raw_data from (n_windows, window, n_tickers) to (n_windows, n_tickers, window)
        full_data[valid_positions, :, :] = self.raw_data.transpose(0, 2, 1)
        
        # Step 3: Reshape to 2D format efficiently 
        # (n_times * n_tickers, window) for DataFrame construction
        data_2d = full_data.reshape(-1, window)
        
        # Step 4: Create index arrays efficiently using broadcasting
        time_repeated = np.repeat(self.time_index, n_tickers)
        ticker_tiled = np.tile(self.ticker_columns, len(self.time_index))
        
        # Step 5: Create column names once
        column_names = [f'window_{i}' for i in range(window)]
        
        # Step 6: Direct DataFrame creation with MultiIndex
        resid_df = pd.DataFrame(
            data_2d,
            columns=column_names,
            index=pd.MultiIndex.from_arrays(
                [time_repeated, ticker_tiled],
                names=["trade_date", "ticker"]
            )
        )
        
        # Step 7: Sort index efficiently
        resid_df = resid_df.sort_index()
        
        # print(f"✅ Residual structure built: {resid_df.shape}")
        return resid_df
    
    def _build_resid_structure_chunked(self):
        """Chunked processing for very large residual datasets."""
        n_windows, window, n_tickers = self.raw_data.shape
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 tickers at a time
        results = []
        
        for chunk_start in range(0, n_tickers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_tickers)
            print(f"  Processing tickers {chunk_start}-{chunk_end-1}")
            
            # Extract chunk
            chunk_data = self.raw_data[:, :, chunk_start:chunk_end]
            chunk_tickers = self.ticker_columns[chunk_start:chunk_end]
            
            # Process chunk using standard method
            chunk_result = self._process_resid_chunk(chunk_data, chunk_tickers)
            results.append(chunk_result)
        
        # Concatenate all chunks
        final_result = pd.concat(results, axis=0)
        # print(f"✅ Chunked residual structure built: {final_result.shape}")
        return final_result
    
    def _process_resid_chunk(self, chunk_data, chunk_tickers):
        """Process a single chunk of residual data."""
        n_windows, window, chunk_size = chunk_data.shape
        window_dates = self.time_index[self.window-1:self.window-1+n_windows]
        
        # Generate indices for this chunk
        time_idx, ticker_idx, window_idx = np.mgrid[0:n_windows, 0:chunk_size, 0:window]
        
        time_values = window_dates[time_idx.ravel()]
        ticker_values = np.array(chunk_tickers)[ticker_idx.ravel()]
        window_values = window_idx.ravel()
        data_values = chunk_data.ravel()
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_arrays(
            [time_values, ticker_values, window_values],
            names=['trade_date', 'ticker', 'window']
        )
        
        # Create and process chunk
        chunk_series = pd.Series(data_values, index=multi_index, name='residual')
        chunk_unstacked = chunk_series.unstack('window')
        
        # Create target for this chunk
        chunk_target = pd.MultiIndex.from_product(
            [self.time_index, chunk_tickers],
            names=['trade_date', 'ticker']
        )
        
        chunk_aligned = chunk_unstacked.reindex(chunk_target)
        return chunk_aligned.stack(dropna=False).to_frame('resid')
    
    def _build_2d_structure(self):
        """Build standard 2D DataFrame structure."""
        if self.raw_data.ndim == 3:
            # Extract relevant data (e.g., last time step for each window)
            n_windows, _, n_tickers = self.raw_data.shape
            data_2d = np.full((len(self.time_index), n_tickers), np.nan, dtype=np.float64)
            
            for t in range(n_windows):
                data_2d[t + self.window - 1] = self.raw_data[t, 0, :]
            
            return pd.DataFrame(data_2d, index=self.time_index, columns=self.ticker_columns)
        else:
            return pd.DataFrame(self.raw_data, index=self.time_index, columns=self.ticker_columns)
    
    def _build_multi_factor_structure(self):
        """Build multi-factor structure with 2-level MultiIndex and factors as columns - Optimized with full index alignment."""
        n_windows, n_factors, n_tickers = self.raw_data.shape
        
        # print(f"🔄 Building multi-factor structure: {len(self.time_index)}×{n_factors}×{n_tickers}")
        
        # PERFORMANCE OPTIMIZED: Use vectorized operations with full index alignment
        
        # Step 1: Create full-size result array with NaN
        full_data = np.full((len(self.time_index), n_tickers, n_factors), np.nan, dtype=np.float64)
        
        # Step 2: Fill in valid data using vectorized assignment
        # Map window_dates to full time index positions
        window_dates = self.time_index[self.window-1:self.window-1+n_windows]
        valid_positions = np.searchsorted(self.time_index, window_dates)
        
        # Vectorized assignment: full_data[valid_positions, :, :] = raw_data.transpose(0, 2, 1)
        # Need to transpose raw_data from (n_windows, n_factors, n_tickers) to (n_windows, n_tickers, n_factors)
        full_data[valid_positions, :, :] = self.raw_data.transpose(0, 2, 1)
        
        # Step 3: Reshape to 2D format efficiently 
        # (n_times * n_tickers, n_factors) for DataFrame construction
        data_2d = full_data.reshape(-1, n_factors)
        
        # Step 4: Create index arrays efficiently using broadcasting
        time_repeated = np.repeat(self.time_index, n_tickers)
        ticker_tiled = np.tile(self.ticker_columns, len(self.time_index))
        
        # Step 5: Direct DataFrame creation with MultiIndex
        beta_df = pd.DataFrame(
            data_2d,
            columns=self.factor_exprs,
            index=pd.MultiIndex.from_arrays(
                [time_repeated, ticker_tiled],
                names=["trade_date", "ticker"]
            )
        )
        
        # Step 6: Sort index efficiently
        beta_df = beta_df.sort_index()
        
        # print(f"✅ Multi-factor structure built: {beta_df.shape}")
        return beta_df


# Memory pool for array reuse
_MEMORY_POOL = {}

def get_cached_array(shape, dtype=np.float64, fill_value=np.nan):
    """Get cached array from memory pool to avoid repeated allocations."""
    key = (shape, dtype)
    if key in _MEMORY_POOL:
        arr = _MEMORY_POOL[key]
        if arr.shape == shape and arr.dtype == dtype:
            arr.fill(fill_value)
            return arr
    
    # Create new array and cache it
    arr = np.full(shape, fill_value, dtype=dtype)
    _MEMORY_POOL[key] = arr
    return arr.copy()  # Return copy to avoid conflicts


def convert_to_2d_format(raw_results, time_index, ticker_columns, window):
    """Convert raw results to standard 2D DataFrame format."""
    if raw_results.ndim == 3:
        # Extract the last time step for each window
        n_windows, _, n_tickers = raw_results.shape
        res_data = np.full((len(time_index), n_tickers), np.nan, dtype=np.float64)
        
        for t in range(n_windows):
            res_data[t + window - 1] = raw_results[t, 0, :]  # Take first factor or alpha
        
        return pd.DataFrame(res_data, index=time_index, columns=ticker_columns)
    else:
        return pd.DataFrame(raw_results, index=time_index, columns=ticker_columns)


def has_numba():
    """Check if Numba is available for optimization."""
    try:
        import numba
        return True
    except ImportError:
        return False


##############################################################
# Manipulate DataFrame

def split_dataframe_by_rows(df, n_splits):
    """
    Split a DataFrame into chunks by rows equally
    
    Parameters:
        df: The DataFrame to be split
        n_splits: Number of chunks to split into
        
    Returns:
        A list of split DataFrames
    """
    if n_splits <= 0:
        raise ValueError("Number of splits must be a positive integer")
    
    # Use numpy's array_split, which handles unequal splits automatically
    # The first few chunks will have one extra row if not perfectly divisible
    split_dfs = np.array_split(df, n_splits)
    return split_dfs


def split_dataframe_by_chunksize(df, chunk_size):
    """
    Split a DataFrame into chunks with specified row count per chunk
    
    Parameters:
        df: The DataFrame to be split
        chunk_size: Number of rows per chunk
        
    Returns:
        A list of split DataFrames
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    
    # Calculate total number of chunks
    n_chunks = (len(df) + chunk_size - 1) // chunk_size  # Round up
    return [df[i*chunk_size : (i+1)*chunk_size] for i in range(n_chunks)]
