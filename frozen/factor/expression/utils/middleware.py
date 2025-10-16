import numpy as np
import pandas as pd

from typing import List
from ..base import Factor
from .numba_middleware import *


def time_weighted(window, half_life):
    w = 0.5 ** (np.arange(window, 0, -1) / half_life)
    w /= w.sum()
    return w


def rolling(data: pd.DataFrame, window: int) -> np.array:
    clean_data = data.select_dtypes(
        include=[np.number]
    ).astype(np.float64)

    padded_data = np.pad(
        clean_data.to_numpy(),
        ((window - 1, 0), (0, 0)),
        mode="constant",
        constant_values=np.nan
    )

    num_cols = clean_data.shape[1]

    roll = np.squeeze(
        np.lib.stride_tricks.sliding_window_view(
            padded_data, (window, num_cols)), axis=1)
    return roll


def expanding(data: pd.DataFrame) -> np.array:
    """Create expanding windows using vectorized operations"""
    clean_data = data.select_dtypes(
        include=[np.number]
    ).astype(np.float64)
    num_rows, num_cols = clean_data.shape
    result = np.empty((num_rows, num_rows, num_cols))
    result[:] = np.nan
    
    # Vectorized expanding window creation
    data_array = clean_data.to_numpy()
    
    # For each time step i, fill result[i, :i+1, :] with data[:i+1, :]
    for i in range(num_rows):
        result[i, :i+1, :] = data_array[:i+1, :]
    
    return result


def create_sliding_windows_vectorized(arr, window_size):
    """Vectorized sliding window creation using stride_tricks, preserving leading NaNs"""
    n_times, n_tickers = arr.shape
    n_windows = n_times - window_size + 1
    
    # Create result array with full time dimension, filled with NaN
    result = np.full((n_times, window_size, n_tickers), np.nan)
    
    if n_windows > 0:
        shape = (n_windows, window_size, n_tickers)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        sliding_windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        
        # Fill valid windows starting from window_size-1 index
        result[window_size-1:] = sliding_windows
    
    return result


def weighted_var(arr, weights=None):
    w_mean = np.average(arr, axis=0, weights=weights)
    w_var = np.average((arr - w_mean)**2, axis=0, weights=weights)
    return w_var


def weighted_std(arr, weights=None):
    w_var = weighted_var(arr, weights)
    w_std = np.sqrt(w_var)
    return w_std


def weighted_cov(x, y, weights=None):
    assert x.shape == y.shape
    mu_x = np.average(x, axis=0, weights=weights)
    mu_y = np.average(y, axis=0, weights=weights)
    cov_w = np.average((x - mu_x) * (y - mu_y), axis=0, weights=weights)
    return cov_w


def weighted_corr(x, y, weights=None):
    assert x.shape == y.shape
    cov = weighted_cov(x, y, weights)
    var_x = weighted_var(x, weights)
    var_y = weighted_var(y, weights)
    corr = cov / np.sqrt(var_x * var_y)
    return corr


def cumsum_extremes(data: pd.DataFrame, window: int, direction='forward'):
    
    if direction == 'backward':
        reversed_data = data.iloc[::-1]
        max_rev, min_rev = cumsum_extremes(reversed_data, window, direction='forward')
        return max_rev.iloc[::-1], min_rev.iloc[::-1]
    
    cumsum = data.cumsum()
    
    roll_max = cumsum.rolling(window).max()
    roll_min = cumsum.rolling(window).min()
    
    shifted = cumsum.shift(window).fillna(0)
    
    max_result = roll_max - shifted
    min_result = roll_min - shifted

    if window > 1:
        initial_max = cumsum.expanding().max().iloc[:window-1]
        initial_min = cumsum.expanding().min().iloc[:window-1]
        
        max_result.iloc[:window-1] = initial_max
        min_result.iloc[:window-1] = initial_min
    
    return max_result, min_result


def agg_by_window(data, window, agg_func="mean", min_periods=None, label="right", cal_rule="calendar"):
    """
    Aggregates time series data using fixed window based on data start date.
    Automatically drops the last incomplete window.
    
    Groups data points into consecutive fixed-length windows, applies the specified 
    aggregation function to each window, and returns the aggregated results with 
    appropriate window labels.
    
    Parameters:
    -----------
        data : pandas DataFrame
            Time series data with datetime index to be aggregated
        window : int
            Size of the fixed window in natural days or trading days
        agg_func : str or callable, default "mean"
            Aggregation function to apply to each window. 
            Built-in options: 'mean', 'sum', 'last', 'first'; or a custom function
        min_periods : int, optional
            Minimum number of valid data points required in a window to compute 
            the aggregation. If not met, the result for that window will be NaN.
            Defaults to None, which requires a full window of data points.
        label : str, default "right"
            Position of the label for each aggregated window:
            - 'right': Use the last date of the window
            - 'left': Use the first date of the window
            - 'center': Use the middle date of the window
        cal_rule : str, default "calendar"
            Calendar rule for block split by date
            - 'calendar': Use calendar days for grouping
            - 'trade': Use trade days for grouping
    """
    start_date = data.index.min()
    end_date = data.index.max()
    
    if cal_rule == "calendar":
        window_boundaries = pd.date_range(
            start=start_date, 
            end=end_date + pd.Timedelta(days=window), 
            freq=f'{window}D'
        )
    elif cal_rule == "trade":
        trading_days = data.index.normalize().unique().sort_values()
        window_boundaries = trading_days[::window]
        if trading_days[-1] not in window_boundaries:
            # Add the day AFTER the last trading day to include the last trading day in the window
            window_boundaries = window_boundaries.append(pd.DatetimeIndex([trading_days[-1] + pd.Timedelta(days=1)]))
        else:
            # If the last trading day is a boundary, extend it by one day to include it
            window_boundaries = window_boundaries[:-1].append(pd.DatetimeIndex([trading_days[-1] + pd.Timedelta(days=1)]))
    else:
        raise ValueError(f"Invalid type of calendar rule: {cal_rule}")
    
    bins = list(zip(window_boundaries[:-1], window_boundaries[1:]))
    
    labels = []
    for start, end in bins:
        if label == "right":
            labels.append(end - pd.Timedelta(days=1))
        elif label == "left":
            labels.append(start)
        elif label == "center":
            mid_point = start + (end - start) / 2
            labels.append(mid_point)
    
    group_assignments = pd.cut(
        data.index,
        bins=pd.DatetimeIndex([b[0] for b in bins] + [bins[-1][1]]),
        labels=labels,
        right=False  # [start, end)
    )
    
    grouped = data.groupby(group_assignments)
    
    if callable(agg_func):
        agg_data = grouped.apply(
            lambda g: g.agg(agg_func) if (min_periods is None or len(g) >= min_periods) else np.nan
        )
    else:
        agg_data = grouped.agg(agg_func)
        
        if min_periods is not None:
            counts = grouped.count()
            agg_data = agg_data.where(counts >= min_periods, np.nan)
    
    if not bins:
        return agg_data
    
    last_bin = bins[-1]
    
    if cal_rule == "calendar":
        # Check if the last window has sufficient data points for a complete window
        last_window_data_count = ((data.index >= last_bin[0]) & 
                                  (data.index < last_bin[1])).sum()
        # Drop the last window if it has fewer data points than the window size
        if last_window_data_count < window:
            if not agg_data.empty:
                agg_data = agg_data.iloc[:-1]
    else:  # trade
        actual_days = ((data.index >= last_bin[0]) & 
                       (data.index < last_bin[1])).sum()
        if actual_days < window:
            if not agg_data.empty:
                agg_data = agg_data.iloc[:-1]
    
    return agg_data


def agg_by_resampler(data, freq, agg_func="mean", min_periods=None, label="right"):

    if callable(agg_func):
        agg_data = data.resample(freq).apply(
            lambda g: g.agg(agg_func) if min_periods is None or len(g) >= min_periods else np.nan
        )
    else:
        agg_data = data.resample(freq, label=label).agg(agg_func)
    
    if agg_data.index[-1] > data.index[-1]:
        agg_data = agg_data.iloc[:-1] 
    
    return agg_data


def general_linreg_2D_standard(y, x: List[np.ndarray], weights=None, param="beta"):
    """
    2-D Linear Regression, with several dependent variable Ys 
    regress against the same set of Xs

    Supports both single-factor and multi-factor Ordinary Least 
    Squares (OLS) and Weighted Least Squares (WLS) regression
    
    Optimized vectorized implementation for better performance.
    """
    n_times, n_tickers = y.shape
    
    # Expand dimensions for all x to match y if needed
    for i, xi in enumerate(x):
        if xi.shape[1] == 1 and n_tickers > 1:
            x[i] = np.tile(xi, n_tickers)
    
    # Stack all x variables and add intercept
    x_stack = np.stack(x, axis=-1)  # Shape: (n_times, n_tickers, n_factors)
    ones = np.ones((n_times, n_tickers, 1))
    X = np.concatenate([x_stack, ones], axis=-1)  # Shape: (n_times, n_tickers, n_factors+1)
    
    # Initialize result arrays
    if param == "beta":
        result = np.full((len(x), n_tickers), np.nan, dtype=np.float64)
    elif param == "alpha":
        result = np.full((1, n_tickers), np.nan, dtype=np.float64)
    elif param == "resid":
        result = np.full((n_times, n_tickers), np.nan, dtype=np.float64)
    
    # Process all tickers at once using vectorized operations
    Y = y  # Shape: (n_times, n_tickers)
    
    # Create valid data mask
    valid_mask = ~np.isnan(Y)
    # Vectorized mask creation: check all factors at once
    valid_mask &= ~np.isnan(X).any(axis=2)
    
    # Process each ticker (still need loop for different valid masks per ticker)
    for ticker in range(n_tickers):
        ticker_mask = valid_mask[:, ticker]
        if ticker_mask.sum() < 2:
            continue
            
        X_valid = X[ticker_mask, ticker, :]  # Shape: (n_valid, n_factors+1)
        Y_valid = Y[ticker_mask, ticker]     # Shape: (n_valid,)
        
        if weights is not None:
            w_valid = weights[ticker_mask]
            W = np.sqrt(np.diag(w_valid))
            X_w = W @ X_valid
            Y_w = W @ Y_valid
        else:
            X_w, Y_w = X_valid, Y_valid
        
        try:
            # Use analytic solution: beta = (X'X)^-1 X'y
            XtX = X_w.T @ X_w
            XtY = X_w.T @ Y_w
            coef = np.linalg.solve(XtX, XtY)
            
            if param == "beta":
                result[:, ticker] = coef[:-1]
            elif param == "alpha":
                result[0, ticker] = coef[-1]
            elif param == "resid":
                resid = Y_w - X_w @ coef
                result[ticker_mask, ticker] = resid
                
        except (np.linalg.LinAlgError, ValueError):
            # Handle singular matrices
            continue
    
    return result


def general_linreg_2D_vectorized(y, x: List[np.ndarray], weights=None, param="beta"):
    """
    Fully vectorized 2-D Linear Regression for maximum performance.
    
    This version processes all tickers simultaneously using full vectorization.
    """
    n_times, n_tickers = y.shape
    
    # Expand dimensions for all x to match y if needed
    for i, xi in enumerate(x):
        if xi.shape[1] == 1 and n_tickers > 1:
            x[i] = np.tile(xi, n_tickers)
    
    # Stack all x variables and add intercept
    x_stack = np.stack(x, axis=-1)  # Shape: (n_times, n_tickers, n_factors)
    ones = np.ones((n_times, n_tickers, 1))
    X = np.concatenate([x_stack, ones], axis=-1)  # Shape: (n_times, n_tickers, n_factors+1)
    
    # Initialize result arrays
    if param == "beta":
        result = np.full((len(x), n_tickers), np.nan, dtype=np.float64)
    elif param == "alpha":
        result = np.full((1, n_tickers), np.nan, dtype=np.float64)
    elif param == "resid":
        result = np.full((n_times, n_tickers), np.nan, dtype=np.float64)
    
    Y = y  # Shape: (n_times, n_tickers)
    
    # Create valid data mask
    valid_mask = ~np.isnan(Y)
    # Vectorized mask creation: check all factors at once
    valid_mask &= ~np.isnan(X).any(axis=2)
    
    # Check if we can process all tickers with the same valid mask (required for vectorized)
    if not np.all(valid_mask == valid_mask[:, [0]]):
        raise RuntimeError("Cannot use vectorized mode: tickers have different valid data patterns")
    
    # All tickers have the same valid data pattern - fully vectorized
    common_mask = valid_mask[:, 0]
    if common_mask.sum() < 2:
        raise RuntimeError("Insufficient valid data points for regression")
        
    X_valid = X[common_mask]  # Shape: (n_valid, n_tickers, n_factors+1)
    Y_valid = Y[common_mask]  # Shape: (n_valid, n_tickers)
    
    if weights is not None:
        w_valid = weights[common_mask]
        # Apply weights to each ticker
        W_sqrt = np.sqrt(w_valid)
        X_w = X_valid * W_sqrt[:, np.newaxis, np.newaxis]
        Y_w = Y_valid * W_sqrt[:, np.newaxis]
    else:
        X_w, Y_w = X_valid, Y_valid
    
    # Vectorized computation for all tickers
    # X_w shape: (n_valid, n_tickers, n_factors+1)  
    # Y_w shape: (n_valid, n_tickers)
    
    # For matrix multiplication X.T @ X and X.T @ Y, we need to compute:
    # For each ticker j: XtX[j, k, l] = sum_over_i(X_w[i, j, k] * X_w[i, j, l])
    # For each ticker j: XtY[j, k] = sum_over_i(X_w[i, j, k] * Y_w[i, j])
    
    # XtX shape: (n_tickers, n_factors+1, n_factors+1) 
    XtX = np.einsum('ijk,ijl->jkl', X_w, X_w)
    # XtY shape: (n_tickers, n_factors+1)
    XtY = np.einsum('ijk,ij->jk', X_w, Y_w)
    
    # Solve for all tickers at once
    coef = np.linalg.solve(XtX, XtY)  # Shape: (n_tickers, n_factors+1)
    
    if param == "beta":
        result[:, :] = coef[:, :-1].T  # coef shape: (n_tickers, n_factors) -> (n_factors, n_tickers)
    elif param == "alpha":
        result[0, :] = coef[:, -1]  # coef shape: (n_tickers,)
    elif param == "resid":
        # Compute residuals for all tickers
        # For each ticker j: Y_pred[i, j] = sum_over_k(X_w[i, j, k] * coef[j, k])
        Y_pred = np.einsum('ijk,jk->ij', X_w, coef)  # shape: (n_valid, n_tickers)
        resid = Y_w - Y_pred  # shape: (n_valid, n_tickers)
        result[common_mask] = resid
        
    return result


def ts_linreg_standard(y, x_factors, y_array, x_arrays, window, weights, param,
                       time_index, ticker_columns, x_expr_str, is_multi_factor):
    """Standard ts_linreg implementation for smaller datasets."""
    
    n_times, n_tickers = y_array.shape
    
    if param == "resid":
        # Handle residuals with 2-level MultiIndex and window as columns - OPTIMIZED for index alignment
        # Create full-size result structure efficiently using vectorized operations
        
        # Step 1: Create full-size result array with NaN
        full_data = np.full((len(time_index), n_tickers, window), np.nan, dtype=np.float64)
        
        # Step 2: Fill in valid results using vectorized assignment
        for t in range(n_times - window + 1):
            y_window = y_array[t:t + window]
            x_window = [x_arr[t:t + window] for x_arr in x_arrays]
            resid = general_linreg_2D_standard(y_window, x_window, weights, param)
            
            # Vectorized assignment to the correct position
            time_pos = t + window - 1
            full_data[time_pos, :, :] = resid.T  # Transpose to match (n_tickers, window) shape
        
        # Step 3: Reshape to 2D format efficiently 
        data_2d = full_data.reshape(-1, window)
        
        # Step 4: Create index arrays efficiently
        time_repeated = np.repeat(time_index, n_tickers)
        ticker_tiled = np.tile(ticker_columns, len(time_index))
        column_names = [f'window_{i}' for i in range(window)]
        
        # Step 5: Direct DataFrame creation
        resid_df = pd.DataFrame(
            data_2d,
            columns=column_names,
            index=pd.MultiIndex.from_arrays(
                [time_repeated, ticker_tiled],
                names=["trade_date", "ticker"]
            )
        )
        resid_df = resid_df.sort_index()
        
        res_expr = f"Resid({y.expr}, {x_expr_str}, {window})"
        return Factor(resid_df, res_expr)
        
    elif param == "beta":
        if is_multi_factor:
            # Multi-factor beta processing - OPTIMIZED for index alignment
            # Create full-size result structure efficiently using vectorized operations
            
            # Step 1: Create full-size result array with NaN
            full_data = np.full((len(time_index), n_tickers, len(x_factors)), np.nan, dtype=np.float64)
            
            # Step 2: Fill in valid results using vectorized assignment
            for t in range(n_times - window + 1):
                y_window = y_array[t:t + window]
                x_window = [x_arr[t:t + window] for x_arr in x_arrays]
                beta_result = general_linreg_2D_standard(y_window, x_window, weights, param)
                
                # Vectorized assignment: beta_result shape is (n_factors, n_tickers)
                time_pos = t + window - 1
                full_data[time_pos, :, :] = beta_result.T  # Transpose to (n_tickers, n_factors)
            
            # Step 3: Reshape to 2D format efficiently 
            data_2d = full_data.reshape(-1, len(x_factors))
            
            # Step 4: Create index arrays efficiently
            time_repeated = np.repeat(time_index, n_tickers)
            ticker_tiled = np.tile(ticker_columns, len(time_index))
            x_exprs = [xf.expr for xf in x_factors]
            
            # Step 5: Direct DataFrame creation
            beta_df = pd.DataFrame(
                data_2d,
                columns=x_exprs,
                index=pd.MultiIndex.from_arrays(
                    [time_repeated, ticker_tiled],
                    names=["trade_date", "ticker"]
                )
            )
            beta_df = beta_df.sort_index()
            
            res_expr = f"Beta({y.expr}, {x_expr_str}, {window})"
            return Factor(beta_df, res_expr)
        else:
            # Single factor beta
            res_data = np.full(y.data.shape, np.nan, dtype=np.float64)
            
            for t in range(n_times - window + 1):
                y_window = y_array[t:t + window]
                x_window = [x_arr[t:t + window] for x_arr in x_arrays]
                beta = general_linreg_2D_standard(y_window, x_window, weights, param)
                res_data[t + window - 1] = beta
            
            res_data = pd.DataFrame(res_data, index=time_index, columns=ticker_columns)
            res_expr = f"Beta({y.expr}, {x_expr_str}, {window})"
            return Factor(res_data, res_expr)
        
    elif param == "alpha":
        # Alpha processing
        res_data = np.full(y.data.shape, np.nan, dtype=np.float64)
        
        for t in range(n_times - window + 1):
            y_window = y_array[t:t + window]
            x_window = [x_arr[t:t + window] for x_arr in x_arrays]
            alpha = general_linreg_2D_standard(y_window, x_window, weights, param)
            res_data[t + window - 1] = alpha
        
        res_data = pd.DataFrame(res_data, index=time_index, columns=ticker_columns)
        res_expr = f"Alpha({y.expr}, {x_expr_str}, {window})"
        return Factor(res_data, res_expr)
        
    else:
        raise ValueError(f"Unsupported param: {param}")


def ts_linreg_vectorized(y, x_factors, y_array, x_arrays, window, weights, param, time_index, ticker_columns, x_expr_str, is_multi_factor):
    """Vectorized batch processing implementation using stride_tricks for sliding windows."""
    
    n_times, n_tickers = y_array.shape
    
    # Create sliding windows using stride_tricks for all arrays at once
    def create_sliding_windows_3d(arr, window_size):
        """Create sliding windows using stride_tricks"""
        if arr.ndim == 2:
            n_times, n_tickers = arr.shape
            n_windows = n_times - window_size + 1
            if n_windows <= 0:
                return np.array([])
            shape = (n_windows, window_size, n_tickers)
            strides = (arr.strides[0], arr.strides[0], arr.strides[1])
            return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        return arr
    
    # Create sliding windows for all arrays
    y_windows = create_sliding_windows_3d(y_array, window)  # shape: (n_windows, window, n_tickers)
    x_windows = [create_sliding_windows_3d(x_arr, window) for x_arr in x_arrays]  # list of (n_windows, window, n_tickers)
    
    if len(y_windows) == 0:  # Handle edge case where window is larger than data
        if param == "resid":
            full_data = np.full((len(time_index), len(ticker_columns), window), np.nan, dtype=np.float64)
            data_2d = full_data.reshape(-1, window)
            time_repeated = np.repeat(time_index, len(ticker_columns))
            ticker_tiled = np.tile(ticker_columns, len(time_index))
            column_names = [f'window_{i}' for i in range(window)]
            resid_df = pd.DataFrame(
                data_2d,
                columns=column_names,
                index=pd.MultiIndex.from_arrays([time_repeated, ticker_tiled], names=["trade_date", "ticker"])
            )
            resid_df = resid_df.sort_index()
            res_expr = f"Resid({y.expr}, {x_expr_str}, {window})"
            return Factor(resid_df, res_expr)
        else:
            res_data = np.full(y.data.shape, np.nan, dtype=np.float64)
            res_data = pd.DataFrame(res_data, index=time_index, columns=ticker_columns)
            if param == "beta":
                res_expr = f"Beta({y.expr}, {x_expr_str}, {window})"
            else:
                res_expr = f"Alpha({y.expr}, {x_expr_str}, {window})"
            return Factor(res_data, res_expr)
    
    n_windows = y_windows.shape[0]
    
    # Initialize result array
    if param == "resid":
        # Handle residuals - OPTIMIZED for index alignment
        # Create full-size result structure efficiently using vectorized operations
        
        # Step 1: Create full-size result array with NaN
        full_data = np.full((len(time_index), len(ticker_columns), window), np.nan, dtype=np.float64)
        
        # Step 2: Process all windows at once using vectorized operations
        for t in range(n_windows):
            y_window = y_windows[t]  # shape: (window, n_tickers)
            x_window = [x_w[t] for x_w in x_windows]  # list of (window, n_tickers)
            resid = general_linreg_2D_vectorized(y_window, x_window, weights, param)
            
            # Vectorized assignment to the correct position
            time_pos = t + window - 1
            full_data[time_pos, :, :] = resid.T  # Transpose to match (n_tickers, window) shape
        
        # Step 3: Reshape and create DataFrame efficiently
        data_2d = full_data.reshape(-1, window)
        time_repeated = np.repeat(time_index, len(ticker_columns))
        ticker_tiled = np.tile(ticker_columns, len(time_index))
        column_names = [f'window_{i}' for i in range(window)]
        
        resid_df = pd.DataFrame(
            data_2d,
            columns=column_names,
            index=pd.MultiIndex.from_arrays([time_repeated, ticker_tiled], names=["trade_date", "ticker"])
        )
        resid_df = resid_df.sort_index()
        
        res_expr = f"Resid({y.expr}, {x_expr_str}, {window})"
        return Factor(resid_df, res_expr)
        
    else:
        # Beta and alpha processing using rolling windows with vectorized computation
        if param == "beta" and is_multi_factor:
            # Multi-factor beta processing - OPTIMIZED for index alignment
            # Create full-size result structure efficiently using vectorized operations
            
            # Step 1: Create full-size result array with NaN
            full_data = np.full((len(time_index), len(ticker_columns), len(x_factors)), np.nan, dtype=np.float64)
            
            # Step 2: Process all windows at once using vectorized operations
            for t in range(n_windows):
                y_window = y_windows[t]  # shape: (window, n_tickers)
                x_window = [x_w[t] for x_w in x_windows]  # list of (window, n_tickers)
                result = general_linreg_2D_vectorized(y_window, x_window, weights, param)
                
                # Vectorized assignment: result shape is (n_factors, n_tickers)
                time_pos = t + window - 1
                full_data[time_pos, :, :] = result.T  # Transpose to (n_tickers, n_factors)
            
            # Step 3: Reshape and create DataFrame efficiently
            data_2d = full_data.reshape(-1, len(x_factors))
            time_repeated = np.repeat(time_index, len(ticker_columns))
            ticker_tiled = np.tile(ticker_columns, len(time_index))
            x_exprs = [xf.expr for xf in x_factors]
            
            beta_df = pd.DataFrame(
                data_2d,
                columns=x_exprs,
                index=pd.MultiIndex.from_arrays([time_repeated, ticker_tiled], names=["trade_date", "ticker"])
            )
            beta_df = beta_df.sort_index()
            
            res_expr = f"Beta({y.expr}, {x_expr_str}, {window})"
            return Factor(beta_df, res_expr)
        else:
            # Single factor beta or alpha processing
            res_data = np.full(y.data.shape, np.nan, dtype=np.float64)
            
            # Process all windows at once using vectorized operations
            for t in range(n_windows):
                y_window = y_windows[t]  # shape: (window, n_tickers)
                x_window = [x_w[t] for x_w in x_windows]  # list of (window, n_tickers)
                
                result = general_linreg_2D_vectorized(y_window, x_window, weights, param)
                res_data[t + window - 1] = result
            
            res_data = pd.DataFrame(res_data, index=time_index, columns=ticker_columns)
            
            if param == "beta":
                res_expr = f"Beta({y.expr}, {x_expr_str}, {window})"
            else:
                res_expr = f"Alpha({y.expr}, {x_expr_str}, {window})"
                
            return Factor(res_data, res_expr)


def cs_mean_by_industry_standard(data, weight, industry_labels):
    """
    Cross-sectional average by industry excluding self with pure pandas operation.
    """
    # Calculate weighted factor
    weighted_factor = data * weight

    # Sum weight and factor by industry groups
    total_weight = weight.groupby(industry_labels, axis=1).sum()
    total_weighted_factor = weighted_factor.groupby(industry_labels, axis=1).sum()
    
    # Map industry sums back to each ticker
    industry_total_weight = total_weight.loc[:, industry_labels]
    industry_total_weighted_factor = total_weighted_factor.loc[:, industry_labels]
    
    # Restore to raw date columns
    industry_total_weight.columns = data.columns
    industry_total_weighted_factor.columns = data.columns
    
    # Exclude self column
    numerator = industry_total_weighted_factor - weighted_factor
    denominator = industry_total_weight - weight
    
    # Deal with zero-division
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], np.nan)
    
    result[denominator == 0] = np.nan
    
    return result


def cs_mean_by_industry_vectorized(data, weight, industry_labels):
    """
    Cross-sectional average by industry excluding self with numpy.
    """
    data_array = data.values
    weight_array = weight.values

    result = np.empty_like(data)
    result[:] = np.nan
    
    # Retrieve unique industry
    unique_industries = np.unique(industry_labels)
    unique_industries = unique_industries[~pd.isnull(unique_industries)]
    
    for industry in unique_industries:
        # Get ticker index within an industry
        mask = (industry_labels == industry)
        if np.sum(mask) < 2:  # skip if only one ticker
            continue
        
        ind_factor = data_array[:, mask]
        ind_cap = weight_array[:, mask]
        
        # Calculate industry weighted sums
        total_weight = np.sum(ind_cap, axis=1, keepdims=True)
        total_weighted_factor = np.sum(ind_factor * ind_cap, axis=1, keepdims=True)
        
        # Exclude self
        numerator = total_weighted_factor - (ind_factor * ind_cap)
        denominator = total_weight - ind_cap
        
        # Deal with zero-division
        with np.errstate(divide='ignore', invalid='ignore'):
            ind_result = numerator / denominator
            ind_result[denominator == 0] = np.nan
        
        result[:, mask] = ind_result
    
    result = pd.DataFrame(result, index=data.index, columns=data.columns)
    return result