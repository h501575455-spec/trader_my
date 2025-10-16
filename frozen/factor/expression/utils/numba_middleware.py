import numpy as np
import pandas as pd
from numba import jit, prange
from ..base import Factor
from .tools import convert_to_2d_format, OptimizedFactor


@jit(nopython=True, cache=True)
def numba_sliding_windows(data, window):
    """Numba-optimized sliding window creation"""
    n_times, n_tickers = data.shape
    n_windows = n_times - window + 1
    
    result = np.full((n_windows, window, n_tickers), np.nan)
    
    for w in range(n_windows):
        for i in range(window):
            for t in range(n_tickers):
                result[w, i, t] = data[w + i, t]
    
    return result


@jit(nopython=True)
def numba_cov(x, y):
    assert x.shape == y.shape
    cov_m = np.zeros(x.shape[1])
    for i in range(len(cov_m)):
        cov_m[i] = np.cov(x[:, i], y[:, i])[0, 1]
    return cov_m


@jit(nopython=True)
def numba_corr(x, y):
    assert x.shape == y.shape
    corr_m = np.zeros(x.shape[1])
    for i in range(len(corr_m)):
        corr_m[i] = np.corrcoef(x[:, i], y[:, i])[0, 1]
    return corr_m


@jit(nopython=True)
def argmax_by_col(data):
    argmax = np.zeros(data.shape[1])
    for i in range(len(argmax)):
        arr = np.copy(data[:, i])
        if np.isnan(arr).all():
            argmax[i] = np.nan
        elif np.isnan(arr).any():
            arr[np.isnan(arr)] = -np.Inf
            # arr = np.where(np.isnan(arr), -np.inf, arr)
            argmax[i] = arr.argmax()
        else:
            argmax[i] = arr.argmax()
    return argmax


@jit(nopython=True)
def numba_ts_argmax(data, window):
    temp = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window - 1:
            temp[i] = np.nan
        else:
            temp[i] = argmax_by_col(data[i-window+1:i+1])
    return temp


@jit(nopython=True)
def mean_by_col(data):
    mean = np.zeros(data.shape[1])
    for i in range(len(mean)):
        arr = np.copy(data[:, i])
        arr = arr[~np.isnan(arr)]
        mean[i] = arr.mean()
        # mean[i] = data[:, i].mean()
    return mean


@jit(nopython=True)
def numba_num_gt_mean(data, window):
    temp = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window - 1:
            temp[i] = np.nan
        else:
            temp[i] = (data[i-window+1:i+1] > mean_by_col(data[i-window+1:i+1])).sum(axis=0)
    return temp


@jit(nopython=True)
def numba_num_lt_mean(data, window):
    temp = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window - 1:
            temp[i] = np.nan
        else:
            temp[i] = (data[i-window+1:i+1] < mean_by_col(data[i-window+1:i+1])).sum(axis=0)
    return temp


@jit(nopython=True)
def numba_lcenter_squared_sum(data, window):
    temp = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window - 1:
            temp[i] = np.nan
        else:
            piece = data[i-window+1:i+1]
            t = (np.where(piece < mean_by_col(piece), piece, np.nan) - mean_by_col(piece)) ** 2
            temp[i] = np.where(~np.isnan(t), t, 0).sum(axis=0)
    return temp


@jit(nopython=True)
def numba_ucenter_squared_sum(data, window):
    temp = np.zeros(data.shape)
    for i in range(len(data)):
        if i < window - 1:
            temp[i] = np.nan
        else:
            piece = data[i-window+1:i+1]
            t = (np.where(piece > mean_by_col(piece), piece, np.nan) - mean_by_col(piece)) ** 2
            temp[i] = np.where(~np.isnan(t), t, 0).sum(axis=0)
    return temp


@jit(nopython=True, parallel=True)
def numba_cumsum_extremes(arr, window, direction='forward'):
    n_times, n_assets = arr.shape
    max_result = np.full((n_times, n_assets), np.nan)
    min_result = np.full((n_times, n_assets), np.nan)
    
    for a in prange(n_assets):
        for t in range(n_times):
            start_idx = max(0, t - window + 1)
            
            current_sum = 0.0
            max_val = -np.inf
            min_val = np.inf
            has_valid = False
            
            if direction == 'forward':
                for i in range(start_idx, t + 1):
                    val = arr[i, a]
                    if not np.isnan(val):
                        current_sum += val
                        has_valid = True
                    if current_sum > max_val:
                        max_val = current_sum
                    if current_sum < min_val:
                        min_val = current_sum
            
            elif direction == 'backward':
                for i in range(t, start_idx - 1, -1):
                    val = arr[i, a]
                    if not np.isnan(val):
                        current_sum += val
                        has_valid = True
                    if current_sum > max_val:
                        max_val = current_sum
                    if current_sum < min_val:
                        min_val = current_sum
            
            if has_valid:
                max_result[t, a] = max_val
                min_result[t, a] = min_val
            else:
                max_result[t, a] = np.nan
                min_result[t, a] = np.nan
    
    return max_result, min_result


@jit(nopython=True)
def numba_matrix_inv_4x4(A):
    """Optimized 4x4 matrix inversion using analytical formula"""
    # Calculate all 2x2 subdeterminants
    s0 = A[0,0] * A[1,1] - A[1,0] * A[0,1]
    s1 = A[0,0] * A[1,2] - A[1,0] * A[0,2] 
    s2 = A[0,0] * A[1,3] - A[1,0] * A[0,3]
    s3 = A[0,1] * A[1,2] - A[1,1] * A[0,2]
    s4 = A[0,1] * A[1,3] - A[1,1] * A[0,3]
    s5 = A[0,2] * A[1,3] - A[1,2] * A[0,3]
    
    c5 = A[2,2] * A[3,3] - A[3,2] * A[2,3]
    c4 = A[2,1] * A[3,3] - A[3,1] * A[2,3] 
    c3 = A[2,1] * A[3,2] - A[3,1] * A[2,2]
    c2 = A[2,0] * A[3,3] - A[3,0] * A[2,3]
    c1 = A[2,0] * A[3,2] - A[3,0] * A[2,2]
    c0 = A[2,0] * A[3,1] - A[3,0] * A[2,1]
    
    # Calculate determinant
    det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    
    if abs(det) < 1e-12:
        return None
        
    inv_det = 1.0 / det
    
    # Calculate inverse matrix elements
    inv_A = np.zeros((4, 4))
    
    inv_A[0,0] = (A[1,1] * c5 - A[1,2] * c4 + A[1,3] * c3) * inv_det
    inv_A[0,1] = (-A[0,1] * c5 + A[0,2] * c4 - A[0,3] * c3) * inv_det
    inv_A[0,2] = (A[3,1] * s5 - A[3,2] * s4 + A[3,3] * s3) * inv_det
    inv_A[0,3] = (-A[2,1] * s5 + A[2,2] * s4 - A[2,3] * s3) * inv_det
    
    inv_A[1,0] = (-A[1,0] * c5 + A[1,2] * c2 - A[1,3] * c1) * inv_det
    inv_A[1,1] = (A[0,0] * c5 - A[0,2] * c2 + A[0,3] * c1) * inv_det
    inv_A[1,2] = (-A[3,0] * s5 + A[3,2] * s2 - A[3,3] * s1) * inv_det
    inv_A[1,3] = (A[2,0] * s5 - A[2,2] * s2 + A[2,3] * s1) * inv_det
    
    inv_A[2,0] = (A[1,0] * c4 - A[1,1] * c2 + A[1,3] * c0) * inv_det
    inv_A[2,1] = (-A[0,0] * c4 + A[0,1] * c2 - A[0,3] * c0) * inv_det
    inv_A[2,2] = (A[3,0] * s4 - A[3,1] * s2 + A[3,3] * s0) * inv_det
    inv_A[2,3] = (-A[2,0] * s4 + A[2,1] * s2 - A[2,3] * s0) * inv_det
    
    inv_A[3,0] = (-A[1,0] * c3 + A[1,1] * c1 - A[1,2] * c0) * inv_det
    inv_A[3,1] = (A[0,0] * c3 - A[0,1] * c1 + A[0,2] * c0) * inv_det
    inv_A[3,2] = (-A[3,0] * s3 + A[3,1] * s1 - A[3,2] * s0) * inv_det
    inv_A[3,3] = (A[2,0] * s3 - A[2,1] * s1 + A[2,2] * s0) * inv_det
    
    return inv_A

@jit(nopython=True)
def numba_cholesky_solve(A, b):
    """Solve Ax=b using Cholesky decomposition for positive definite matrices"""
    n = A.shape[0]
    L = np.zeros((n, n))
    
    # Cholesky decomposition: A = L * L^T
    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Diagonal elements
                sum_sq = 0.0
                for k in range(j):
                    sum_sq += L[i, k] ** 2
                L[i, j] = np.sqrt(A[i, i] - sum_sq)
            else:  # Off-diagonal elements
                sum_prod = 0.0
                for k in range(j):
                    sum_prod += L[i, k] * L[j, k]
                if abs(L[j, j]) < 1e-12:
                    return None
                L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    # Forward substitution: L * y = b
    y = np.zeros(n)
    for i in range(n):
        sum_ly = 0.0
        for j in range(i):
            sum_ly += L[i, j] * y[j]
        y[i] = b[i] - sum_ly
        if abs(L[i, i]) < 1e-12:
            return None
        y[i] /= L[i, i]
    
    # Backward substitution: L^T * x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ltx = 0.0
        for j in range(i + 1, n):
            sum_ltx += L[j, i] * x[j]
        x[i] = y[i] - sum_ltx
        if abs(L[i, i]) < 1e-12:
            return None
        x[i] /= L[i, i]
    
    return x

@jit(nopython=True)
def numba_matrix_inv(A):
    """Optimized numba-compatible matrix inversion for small matrices."""
    n = A.shape[0]
    
    if n == 2:
        # 2x2 matrix inversion
        det = A[0,0] * A[1,1] - A[0,1] * A[1,0]
        if abs(det) < 1e-12:
            return None
        inv_A = np.array([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]]) / det
        return inv_A
    
    elif n == 3:
        # 3x3 matrix inversion
        det = (A[0,0] * (A[1,1] * A[2,2] - A[1,2] * A[2,1]) -
               A[0,1] * (A[1,0] * A[2,2] - A[1,2] * A[2,0]) +
               A[0,2] * (A[1,0] * A[2,1] - A[1,1] * A[2,0]))
        
        if abs(det) < 1e-12:
            return None
            
        inv_A = np.zeros((3, 3))
        inv_A[0,0] = (A[1,1] * A[2,2] - A[1,2] * A[2,1]) / det
        inv_A[0,1] = (A[0,2] * A[2,1] - A[0,1] * A[2,2]) / det
        inv_A[0,2] = (A[0,1] * A[1,2] - A[0,2] * A[1,1]) / det
        inv_A[1,0] = (A[1,2] * A[2,0] - A[1,0] * A[2,2]) / det
        inv_A[1,1] = (A[0,0] * A[2,2] - A[0,2] * A[2,0]) / det
        inv_A[1,2] = (A[0,2] * A[1,0] - A[0,0] * A[1,2]) / det
        inv_A[2,0] = (A[1,0] * A[2,1] - A[1,1] * A[2,0]) / det
        inv_A[2,1] = (A[0,1] * A[2,0] - A[0,0] * A[2,1]) / det
        inv_A[2,2] = (A[0,0] * A[1,1] - A[0,1] * A[1,0]) / det
        return inv_A
    
    elif n == 4:
        # Use optimized 4x4 analytical inversion
        return numba_matrix_inv_4x4(A)
    
    else:
        # For larger matrices, use Cholesky decomposition for positive definite matrices
        # For XtX matrices in regression, this is usually the case
        try:
            # Try to solve A * inv_A = I using Cholesky
            inv_A = np.zeros((n, n))
            I = np.eye(n)
            
            for col in range(n):
                x = numba_cholesky_solve(A, I[:, col])
                if x is None:
                    # Fallback to Gaussian elimination if Cholesky fails
                    return numba_gaussian_elimination(A)
                inv_A[:, col] = x
            
            return inv_A
            
        except:
            # Final fallback to Gaussian elimination
            return numba_gaussian_elimination(A)

@jit(nopython=True)
def numba_gaussian_elimination(A):
    """Fallback Gaussian elimination implementation"""
    n = A.shape[0]
    # Create augmented matrix [A | I]
    aug = np.zeros((n, 2*n))
    aug[:, :n] = A
    for i in range(n):
        aug[i, n+i] = 1.0
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i+1, n):
            if abs(aug[k, i]) > abs(aug[max_row, i]):
                max_row = k
        
        # Swap rows
        if max_row != i:
            for j in range(2*n):
                aug[i, j], aug[max_row, j] = aug[max_row, j], aug[i, j]
        
        # Check for singular matrix
        if abs(aug[i, i]) < 1e-12:
            return None
        
        # Make diagonal element 1
        pivot = aug[i, i]
        for j in range(2*n):
            aug[i, j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = aug[k, i]
                for j in range(2*n):
                    aug[k, j] -= factor * aug[i, j]
    
    # Extract inverse matrix
    inv_A = aug[:, n:]
    return inv_A


@jit(nopython=True, parallel=True, cache=True)
def numba_batch_linreg_core(y_windows, x_windows, valid_masks, weights, param_code):
    """
    Numba-accelerated batch linear regression core computation
    
    Args:
        y_windows: (n_windows, window, n_tickers)
        x_windows: (n_windows, window, n_tickers) 
        valid_masks: (n_windows, window, n_tickers)
        weights: (window,) or None
        param_code: 0=beta, 1=alpha, 2=resid
    
    Returns:
        results: corresponding result arrays
    """
    n_windows, window, n_tickers = y_windows.shape
    
    if param_code == 0:  # beta
        results = np.full((n_windows, 1, n_tickers), np.nan)
    elif param_code == 1:  # alpha  
        results = np.full((n_windows, 1, n_tickers), np.nan)
    else:  # resid
        results = np.full((n_windows, window, n_tickers), np.nan)
    
    # Parallel processing of all windows
    for w in prange(n_windows):
        for ticker in prange(n_tickers):
            mask = valid_masks[w, :, ticker]
            n_valid = np.sum(mask)
            
            if n_valid < 2:
                continue
            
            # Extract valid data
            y_valid = np.zeros(n_valid)
            x_valid = np.zeros(n_valid)
            w_valid = np.zeros(n_valid)
            
            idx = 0
            for i in range(window):
                if mask[i]:
                    y_valid[idx] = y_windows[w, i, ticker]
                    x_col = min(ticker, x_windows.shape[2] - 1)
                    x_valid[idx] = x_windows[w, i, x_col]
                    if weights is not None:
                        w_valid[idx] = weights[i]
                    else:
                        w_valid[idx] = 1.0
                    idx += 1
            
            X = np.zeros((n_valid, 2))
            X[:, 0] = x_valid
            X[:, 1] = 1.0
            
            # Apply weights
            if weights is not None:
                sqrt_w = np.sqrt(w_valid)
                X_w = X * sqrt_w.reshape(-1, 1)
                y_w = y_valid * sqrt_w
            else:
                X_w = X
                y_w = y_valid
            
            # Solve linear equations (X'X)β = X'y
            XtX = np.zeros((2, 2))
            XtY = np.zeros(2)
            
            for i in range(n_valid):
                for j in range(2):
                    XtY[j] += X_w[i, j] * y_w[i]
                    for k in range(2):
                        XtX[j, k] += X_w[i, j] * X_w[i, k]
            
            # Solve coefficients
            det = XtX[0, 0] * XtX[1, 1] - XtX[0, 1] * XtX[1, 0]
            if abs(det) < 1e-12:
                continue
                
            # Manual 2x2 matrix inversion
            inv_det = 1.0 / det
            coef = np.zeros(2)
            coef[0] = inv_det * (XtX[1, 1] * XtY[0] - XtX[0, 1] * XtY[1])  # beta
            coef[1] = inv_det * (-XtX[1, 0] * XtY[0] + XtX[0, 0] * XtY[1]) # alpha
            
            beta_coef = coef[0]
            alpha_coef = coef[1]
            
            # Store results
            if param_code == 0:  # beta
                results[w, 0, ticker] = beta_coef
            elif param_code == 1:  # alpha
                results[w, 0, ticker] = alpha_coef
            else:  # resid
                resid_valid = np.zeros(n_valid)
                for i in range(n_valid):
                    if weights is not None:
                        # For weighted case: resid = y_w - X_w @ coef
                        y_w_i = y_valid[i] * sqrt_w[i]
                        x_w_i = X_w[i, :]  # [x_weighted, 1_weighted]
                        y_pred_w = x_w_i[0] * coef[0] + x_w_i[1] * coef[1]
                        resid_valid[i] = y_w_i - y_pred_w
                    else:
                        # For unweighted case: resid = y - (beta*x + alpha)
                        y_pred = coef[0] * x_valid[i] + coef[1]
                        resid_valid[i] = y_valid[i] - y_pred
                
                # Place residuals back into the full window array
                valid_idx = 0
                for i in range(window):
                    if mask[i]:
                        results[w, i, ticker] = resid_valid[valid_idx]
                        valid_idx += 1
    
    return results


@jit(nopython=True, parallel=True, cache=True)
def numba_batch_linreg_core_multifactor(y_windows, x_windows_list, valid_masks, weights, param_code):
    """
    Numba-accelerated batch linear regression core computation with multi-factor support
    
    Args:
        y_windows: (n_windows, window, n_tickers)
        x_windows_list: list of (n_windows, window, n_tickers) arrays for each factor
        valid_masks: (n_windows, window, n_tickers)
        weights: (window,) or None
        param_code: 0=beta, 1=alpha, 2=resid
    
    Returns:
        results: corresponding result arrays
    """
    n_windows, window, n_tickers = y_windows.shape
    n_factors = len(x_windows_list)
    
    if param_code == 0:  # beta
        results = np.full((n_windows, n_factors, n_tickers), np.nan)
    elif param_code == 1:  # alpha  
        results = np.full((n_windows, 1, n_tickers), np.nan)
    else:  # resid
        results = np.full((n_windows, window, n_tickers), np.nan)
    
    # Parallel processing of all windows
    for w in prange(n_windows):
        for ticker in prange(n_tickers):
            mask = valid_masks[w, :, ticker]
            n_valid = np.sum(mask)
            
            if n_valid < n_factors + 1:  # Need at least n_factors + 1 points for regression
                continue
            
            # Extract valid data
            y_valid = np.zeros(n_valid)
            X_valid = np.zeros((n_valid, n_factors + 1))  # +1 for intercept
            w_valid = np.zeros(n_valid)
            
            idx = 0
            for i in range(window):
                if mask[i]:
                    y_valid[idx] = y_windows[w, i, ticker]
                    # Extract all factors
                    for f in range(n_factors):
                        x_col = min(ticker, x_windows_list[f].shape[2] - 1)
                        X_valid[idx, f] = x_windows_list[f][w, i, x_col]
                    X_valid[idx, n_factors] = 1.0  # intercept
                    
                    if weights is not None:
                        w_valid[idx] = weights[i]
                    else:
                        w_valid[idx] = 1.0
                    idx += 1
            
            # Apply weights if provided
            if weights is not None:
                sqrt_w = np.sqrt(w_valid)
                X_w = X_valid * sqrt_w.reshape(-1, 1)
                y_w = y_valid * sqrt_w
            else:
                X_w = X_valid
                y_w = y_valid
            
            # Solve linear system: (X'X)^-1 X'y
            try:
                XtX = X_w.T @ X_w
                XtY = X_w.T @ y_w
                
                # Use our numba-compatible matrix inversion
                inv_XtX = numba_matrix_inv(XtX)
                if inv_XtX is None:
                    continue
                
                coef = inv_XtX @ XtY
                
            except:
                continue
            
            if param_code == 0:  # beta
                for f in range(n_factors):
                    results[w, f, ticker] = coef[f]
            elif param_code == 1:  # alpha
                results[w, 0, ticker] = coef[n_factors]  # intercept
            else:  # resid
                # Compute residuals
                resid_valid = y_w - X_w @ coef
                
                # Place residuals back into the full window array
                valid_idx = 0
                for i in range(window):
                    if mask[i]:
                        results[w, i, ticker] = resid_valid[valid_idx]
                        valid_idx += 1
    
    return results


def ts_linreg_numba(y, x_factors, y_array, x_arrays, window, weights, param,
                    time_index, ticker_columns, x_expr_str, is_multi_factor):
    """Numba-optimized ts_linreg implementation."""
    
    n_times, _n_tickers = y_array.shape
    
    # Check if window size is valid
    _validate_window_size(window, n_times)
    
    n_windows = n_times - window + 1
    
    # Create sliding windows using Numba
    y_windows = numba_sliding_windows(y_array, window)
    
    # For multi-factor case, use optimized stacked sliding windows
    if is_multi_factor:
        # Ensure all arrays have consistent memory layout
        x_arrays_consistent = []
        for x_arr in x_arrays:
            # Convert to C-contiguous array to ensure consistent layout
            x_arrays_consistent.append(np.ascontiguousarray(x_arr))
        
        # Use optimized stacked factor windows creation
        x_stacked = numba_create_stacked_factor_windows(x_arrays_consistent, window)
        
        # Create validity masks for multi-factor using stacked data
        valid_masks = ~np.isnan(y_windows)
        if x_stacked.size > 0:
            # Check for NaN in any factor
            valid_masks &= ~np.isnan(x_stacked).any(axis=-1)
        
        # Parameter encoding for Numba
        param_map = {"beta": 0, "alpha": 1, "resid": 2}
        param_code = param_map[param]
        
        # Use optimized multi-factor numba core computation
        raw_results = numba_batch_linreg_core_multifactor_optimized(y_windows, x_stacked, valid_masks, weights, param_code)
        
    else:
        # Single factor case - use original implementation
        x_windows = numba_sliding_windows(x_arrays[0], window)
        
        # Create validity masks
        if x_windows.shape[2] == 1 and y_windows.shape[2] > 1:
            x_valid_base = ~np.isnan(x_windows[:, :, 0:1])
            x_valid_broadcast = np.broadcast_to(x_valid_base, y_windows.shape)
            valid_masks = ~np.isnan(y_windows) & x_valid_broadcast
        else:
            valid_masks = ~np.isnan(y_windows) & ~np.isnan(x_windows)
        
        # Parameter encoding for Numba
        param_map = {"beta": 0, "alpha": 1, "resid": 2}
        param_code = param_map[param]
        
        # Use single-factor numba core computation
        raw_results = numba_batch_linreg_core(y_windows, x_windows, valid_masks, weights, param_code)
    
    # Return optimized Factor with lazy Pandas reconstruction
    return create_optimized_linreg_factor(raw_results, param, y, time_index, ticker_columns, x_expr_str, window, n_windows, is_multi_factor, x_factors)


def create_optimized_linreg_factor(raw_results, param, y_factor, time_index, ticker_columns, x_expr_str, window, n_windows, is_multi_factor, x_factors=None):
    """Create optimized Factor with lazy Pandas reconstruction."""
    
    if param == "resid":
        res_expr = f"Resid({y_factor.expr}, {x_expr_str}, {window})"
        return OptimizedFactor(raw_results, res_expr, param, time_index, ticker_columns, window, n_windows, lazy_rebuild=True)
    elif param == "beta":
        res_expr = f"Beta({y_factor.expr}, {x_expr_str}, {window})"
        if is_multi_factor:
            # Extract factor expressions for multi-factor case
            factor_exprs = [xf.expr for xf in x_factors] if x_factors else []
            return OptimizedFactor(raw_results, res_expr, param, time_index, ticker_columns, window, n_windows, lazy_rebuild=True, is_multi_factor=True, factor_exprs=factor_exprs)
        else:
            # For single factor beta, convert to standard 2D format
            if raw_results is not None:
                res_data = convert_to_2d_format(raw_results, time_index, ticker_columns, window)
                return Factor(res_data, res_expr)
            else:
                # Return empty factor if no results
                res_data = pd.DataFrame(index=time_index, columns=ticker_columns)
                return Factor(res_data, res_expr)
    elif param == "alpha":
        res_expr = f"Alpha({y_factor.expr}, {x_expr_str}, {window})"
        if raw_results is not None:
            res_data = convert_to_2d_format(raw_results, time_index, ticker_columns, window)
            return Factor(res_data, res_expr)
        else:
            # Return empty factor if no results
            res_data = pd.DataFrame(index=time_index, columns=ticker_columns)
            return Factor(res_data, res_expr)


def _validate_window_size(window, n_times):
    if window <= 0:
        raise ValueError(f"Window size must be positive, got {window}")
    if window > n_times:
        raise ValueError(f"Window size ({window}) cannot be larger than the number of time periods ({n_times})")
    if window < 2:
        raise ValueError(f"Window size must be at least 2 for linear regression, got {window}")


# ================ OPTIMIZED MULTIFACTOR FUNCTIONS ================

@jit(nopython=True)
def numba_create_stacked_factor_windows(x_arrays, window):
    """Create stacked sliding windows for multiple factors efficiently"""
    n_factors = len(x_arrays)
    if n_factors == 0:
        # Return properly typed empty array
        return np.empty((0, 0, 0, 0), dtype=np.float64)
    
    # Get dimensions from first array
    n_times, n_tickers = x_arrays[0].shape
    n_windows = n_times - window + 1
    
    if n_windows <= 0:
        # Return properly typed empty array with correct dimensions
        return np.empty((0, window, n_tickers, n_factors), dtype=np.float64)
    
    # Pre-allocate stacked array: (n_windows, window, n_tickers, n_factors)
    stacked_windows = np.full((n_windows, window, n_tickers, n_factors), np.nan, dtype=np.float64)
    
    # Fill stacked array efficiently
    for f in range(n_factors):
        x_arr = x_arrays[f]
        # Use manual sliding window creation
        for w in range(n_windows):
            for i in range(window):
                if x_arr.shape[1] == 1:
                    # Broadcast single-column factor (like market index) across all tickers
                    factor_value = x_arr[w + i, 0]
                    for t in range(n_tickers):
                        stacked_windows[w, i, t, f] = factor_value
                else:
                    # Multi-column factor - use corresponding ticker column
                    for t in range(min(n_tickers, x_arr.shape[1])):
                        stacked_windows[w, i, t, f] = x_arr[w + i, t]
    
    return stacked_windows

@jit(nopython=True, parallel=True, cache=True)
def numba_batch_linreg_core_multifactor_optimized(y_windows, x_stacked, valid_masks, weights, param_code):
    """
    Optimized multi-factor batch linear regression using stacked arrays
    
    Args:
        y_windows: (n_windows, window, n_tickers)
        x_stacked: (n_windows, window, n_tickers, n_factors) 
        valid_masks: (n_windows, window, n_tickers)
        weights: (window,) or None
        param_code: 0=beta, 1=alpha, 2=resid
    
    Returns:
        results: corresponding result arrays
    """
    n_windows, window, n_tickers = y_windows.shape
    n_factors = x_stacked.shape[3]
    
    if param_code == 0:  # beta
        results = np.full((n_windows, n_factors, n_tickers), np.nan)
    elif param_code == 1:  # alpha
        results = np.full((n_windows, 1, n_tickers), np.nan)
    else:  # resid
        results = np.full((n_windows, window, n_tickers), np.nan)
    
    # Parallel processing of all windows
    for w in prange(n_windows):
        y_w = y_windows[w]  # Shape: (window, n_tickers)
        x_w = x_stacked[w]  # Shape: (window, n_tickers, n_factors)
        mask_w = valid_masks[w]  # Shape: (window, n_tickers)
        
        # Process each ticker
        for ticker in range(n_tickers):
            valid_mask = mask_w[:, ticker]
            n_valid = np.sum(valid_mask)
            
            if n_valid < max(n_factors + 1, 2):  # Need at least n_factors+1 points
                continue
            
            # Extract valid data for this ticker
            y_valid = np.zeros(n_valid)
            X_valid = np.zeros((n_valid, n_factors + 1))  # +1 for intercept
            
            idx = 0
            for i in range(window):
                if valid_mask[i]:
                    y_valid[idx] = y_w[i, ticker]
                    # Add factor terms first (indices 0 to n_factors-1)
                    for f in range(n_factors):
                        X_valid[idx, f] = x_w[i, ticker, f]
                    # Add intercept term last (index n_factors)
                    X_valid[idx, n_factors] = 1.0
                    idx += 1
            
            # Apply weights if provided
            if weights is not None:
                w_valid = np.zeros(n_valid)
                idx = 0
                for i in range(window):
                    if valid_mask[i]:
                        w_valid[idx] = weights[i]
                        idx += 1
                
                # Apply weights to data
                sqrt_w = np.sqrt(w_valid)
                y_valid = y_valid * sqrt_w
                for i in range(n_valid):
                    for j in range(n_factors + 1):
                        X_valid[i, j] *= sqrt_w[i]
            
            # Solve regression using optimized matrix operations
            XtX = X_valid.T @ X_valid
            XtY = X_valid.T @ y_valid
            
            # Use optimized matrix inversion
            inv_XtX = numba_matrix_inv(XtX)
            if inv_XtX is None:
                continue
            
            coef = inv_XtX @ XtY
            
            if param_code == 0:  # beta
                for f in range(n_factors):
                    results[w, f, ticker] = coef[f]  # Factor coefficients (indices 0 to n_factors-1)
            elif param_code == 1:  # alpha
                results[w, 0, ticker] = coef[n_factors]  # Intercept (index n_factors)
            elif param_code == 2:  # resid
                # Compute residuals
                y_pred = X_valid @ coef
                residuals = y_valid - y_pred
                
                # Map residuals back to full window
                idx = 0
                for i in range(window):
                    if valid_mask[i]:
                        results[w, i, ticker] = residuals[idx]
                        idx += 1
    
    return results