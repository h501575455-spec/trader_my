import numpy as np
import pandas as pd

from .base import Factor
from .utils.middleware import *
from .utils.tools import has_numba, split_dataframe_by_rows


#================================= MATH OPS =================================


def sqrt(x):

    sqrt_data = np.sqrt(x.data)
    sqrt_expr = f"Sqrt({x.expr})"

    return Factor(sqrt_data, sqrt_expr)


def exp(x):

    exp_data = np.exp(x.data)
    exp_expr = f"Exp({x.expr})"

    return Factor(exp_data, exp_expr)


def log(x, base=None):

    if base is None:
        log_data = np.log(x.data)
        log_expr = f"Log({x.expr})"
    
    elif isinstance(base, Factor):
        log_data = np.log(x.data) / np.log(base.data)
        log_expr = f"Log({x.expr}, {base.expr})"
    else:
        log_data = np.log(x.data) / np.log(base)
        log_expr = f"Log({x.expr}, {base})"

    return Factor(log_data, log_expr)


def power(x, power):

    power_data = np.power(x.data, power)
    power_expr = f"Power({x.expr}, {power})"

    return Factor(power_data, power_expr)


def signedpower(x, power):

    sp_data = np.sign(x.data) * np.power(np.abs(x.data), power)
    sp_expr = f"SignedPower({x.expr}, {power})"

    return Factor(sp_data, sp_expr)


def lt(x, y):

    if isinstance(y, Factor):
        lt_data = x.data < y.data
        lt_expr = f"{x.expr} < {y.expr}"
    else:
        lt_data = x.data < y
        lt_expr = f"{x.expr} < {y}"

    return Factor(lt_data, lt_expr)


def gt(x, y):

    if isinstance(y, Factor):
        gt_data = x.data > y.data
        gt_expr = f"{x.expr} > {y.expr}"
    else:
        gt_data = x.data > y
        gt_expr = f"{x.expr} > {y}"

    return Factor(gt_data, gt_expr)


def lte(x, y):

    if isinstance(y, Factor):
        lte_data = x.data <= y.data
        lte_expr = f"{x.expr} <= {y.expr}"
    else:
        lte_data = x.data <= y
        lte_expr = f"{x.expr} <= {y}"

    return Factor(lte_data, lte_expr)


def gte(x, y):

    if isinstance(y, Factor):
        gte_data = x.data >= y.data
        gte_expr = f"{x.expr} >= {y.expr}"
    else:
        gte_data = x.data >= y
        gte_expr = f"{x.expr} >= {y}"

    return Factor(gte_data, gte_expr)


def eq(x, y):

    if isinstance(y, Factor):
        eq_data = x.data == y.data
        eq_expr = f"{x.expr} = {y.expr}"
    else:
        eq_data = x.data == y
        eq_expr = f"{x.expr} = {y}"

    return Factor(eq_data, eq_expr)


def ne(x, y):

    if isinstance(y, Factor):
        ne_data = x.data != y.data
        ne_expr = f"{x.expr} != {y.expr}"
    else:
        ne_data = x.data != y
        ne_expr = f"{x.expr} != {y}"

    return Factor(ne_data, ne_expr)


def add(x, y):

    if isinstance(y, Factor):
        add_data = x.data + y.data
        add_expr = f"Add({x.expr}, {y.expr})"
    else:
        add_data = x.data + y
        add_expr = f"Add({x.expr}, {y})"

    return Factor(add_data, add_expr)


def sub(x, y):

    if isinstance(y, Factor):
        minus_data = x.data - y.data
        minus_expr = f"Sub({x.expr}, {y.expr})"
    else:
        minus_data = x.data - y
        minus_expr = f"Sub({x.expr}, {y})"

    return Factor(minus_data, minus_expr)


def mul(x, y):

    if isinstance(y, Factor):
        mltp_data = x.data * y.data
        mltp_expr = f"Mul({x.expr}, {y.expr})"
    else:
        mltp_data = x.data * y
        mltp_expr = f"Mul({x.expr}, {y})"

    return Factor(mltp_data, mltp_expr)


def div(x, y):

    if isinstance(y, Factor):
        div_data = x.data / y.data
        div_expr = f"Div({x.expr}, {y.expr})"
    else:
        div_data = x.data / y
        div_expr = f"Div({x.expr}, {y})"

    return Factor(div_data, div_expr)


def where(condition, x, y):
    
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        temp = np.where(condition.data, x, y)
        result_expr = f"where({condition.expr} ? {x} : {y})"

    elif isinstance(x, (int, float)) and isinstance(y, Factor):
        temp = np.where(condition.data, x, y.data)
        result_expr = f"where({condition.expr} ? {x} : {y.expr})"
    
    elif isinstance(x, Factor) and isinstance(y, (int, float)):
        temp = np.where(condition.data, x.data, y)
        result_expr = f"where({condition.expr} ? {x.expr} : {y})"
    
    else:
        temp = np.where(condition.data, x.data, y.data)
        result_expr = f"where({condition.expr} ? {x.expr} : {y.expr})"
    
    result_data = pd.DataFrame(temp, index=condition.data.index, columns=condition.data.columns)
    
    return Factor(result_data, result_expr)


#================================= DATAFRAME OPS =================================


def reverse_index(x):
    
    reverse_data = x.data.copy()
    reverse_data = reverse_data[::-1]
    reverse_expr = f"Reverse({x.expr})"

    return Factor(reverse_data, reverse_expr)


def reindex(x, target_index):

    original_name = x.data.index.name
    reindex_data = x.data.reindex(target_index).rename_axis(original_name)
    reindex_expr = f"Reindex({x.expr})"

    return Factor(reindex_data, reindex_expr)


def fillna(x, value=None):

    fillna_data = x.data.fillna(value)
    fillna_expr = f"FillNA({x.expr}, {value})"
    
    return Factor(fillna_data, fillna_expr)


def ffill(x):

    ffill_data = x.data.ffill()
    ffill_expr = f"ForwardFill({x.expr})"

    return Factor(ffill_data, ffill_expr)


def bfill(x):

    bfill_data = x.data.bfill()
    bfill_expr = f"BackwardFill({x.expr})"

    return Factor(bfill_data, bfill_expr)


def copy(x, fill_value=None):

    if fill_value is None:
        copy_data = x.data.copy()
    else:
        copy_data = pd.DataFrame(fill_value, index=x.data.index, columns=x.data.columns)
    
    copy_expr = x.expr

    return Factor(copy_data, copy_expr)


#================================= TIME-SERIES OPS =================================


def delay(x, window):

    delay_data = x.data.shift(window)
    delay_expr = f"Delay({x.expr}, {window})"

    return Factor(delay_data, delay_expr)


def delta(x, window):

    delta_data = x.data - x.data.shift(window)
    delta_expr = f"Delta({x.expr}, {window})"

    return Factor(delta_data, delta_expr)


def ewma(x, window, half_life=None):

    if half_life is None:
        half_life = window // 4
    
    decay = 1 - np.exp(-np.log(2) / half_life)
    ewma_data = x.data.ewm(alpha=decay, adjust=False).mean()
    ewma_expr = f"EWMA({x.expr}, {window})"
    
    return Factor(ewma_data, ewma_expr)


def ts_rank(x, window):

    ranked_data = x.data.rolling(window).rank(ascending=False, method="average")
    ranked_expr = f"Rank({x.expr})"

    return Factor(ranked_data, ranked_expr)


def ts_min(x, window):

    min_data = x.data.rolling(window).min()
    min_expr = f"Min({x.expr}, {window})"

    return Factor(min_data, min_expr)


def ts_max(x, window):

    max_data = x.data.rolling(window).max()
    max_expr = f"Max({x.expr}, {window})"

    return Factor(max_data, max_expr)


def ts_argmax(x, window):
    
    argmax_data = pd.DataFrame(numba_ts_argmax(np.array(x.data), window), index=x.data.index, columns=x.data.columns)
    argmax_expr = f"Ts_ArgMax({x.expr}, {window})"

    return Factor(argmax_data, argmax_expr)


def ts_sum(x, window):
    
    sum_data = x.data.rolling(window).sum()
    sum_expr = f"Sum({x.expr}, {window})"
    
    return Factor(sum_data, sum_expr)


def ts_prod(x, window):

    prod_data = x.data.rolling(window).prod()
    prod_expr = f"Prod({x.expr}, {window})"

    return Factor(prod_data, prod_expr)


def ts_mean(x, window, weight=False, half_life=None):

    if weight is False:
        mean_data = x.data.rolling(window).mean()
    
    else:
        assert half_life is not None, "Parameter half_life must be specified if weight is True."
        decay = time_weighted(window, half_life)

        mean_data = np.zeros(x.data.shape)
        for i, plane in enumerate(rolling(x.data, window)):
            mean_data[i] = np.nan if i < window-1 else np.average(plane, axis=0, weights=decay)
        mean_data = pd.DataFrame(mean_data, index=x.data.index, columns=x.data.columns)
    
    mean_expr = f"Mean({x.expr}, {window})"

    return Factor(mean_data, mean_expr)


def ts_stddev(x, window, weight=False, half_life=None):

    if weight is False:
        std_data = x.data.rolling(window).std()
    
    else:
        assert half_life is not None, "Parameter half_life must be specified if weight is True."
        decay = time_weighted(window, half_life)

        std_data = np.zeros(x.data.shape)
        for i, plane in enumerate(rolling(x.data, window)):
            std_data[i] = np.nan if i < window-1 else weighted_std(plane, weights=decay)
        
        std_data = pd.DataFrame(std_data, index=x.data.index, columns=x.data.columns)
    
    std_expr = f"Std({x.expr}, {window})"

    return Factor(std_data, std_expr)


def ts_cov(y, x, window, weight=False, half_life=None):

    # expand the dimension of X to match the dimension of y
    if x.data.shape[1] == 1 and y.data.shape[1] > 1:
        x.data = pd.concat([x.data] * y.data.shape[1], axis=1)
        x.data.columns = y.data.columns

    cov_data = np.zeros(y.data.shape)
    if weight is False:
        for i, (piece1, piece2) in enumerate(zip(rolling(y.data, window), rolling(x.data, window))):
            cov_data[i] = np.nan if i < window-1 else weighted_cov(np.array(piece1), np.array(piece2))
    else:
        assert half_life is not None, "Parameter half_life must be specified if weight is True."
        decay = time_weighted(window , half_life)

        for i, (piece1, piece2) in enumerate(zip(rolling(y.data, window), rolling(x.data, window))):
            cov_data[i] = np.nan if i < window-1 else weighted_cov(piece1, piece2, weights=decay)
    
    cov_data = pd.DataFrame(cov_data, index=y.data.index, columns=y.data.columns)
    cov_expr = f"Cov({y.expr}, {x.expr}, {window})"

    return Factor(cov_data, cov_expr)


def ts_corr(y, x, window, weight=False, half_life=None):

    # expand the dimension of X to match the dimension of y
    if x.data.shape[1] == 1 and y.data.shape[1] > 1:
        x.data = pd.concat([x.data] * y.data.shape[1], axis=1)
        x.data.columns = y.data.columns

    corr_data = np.zeros(y.data.shape)
    if weight is False:
        for i, (piece1, piece2) in enumerate(zip(rolling(y.data, window), rolling(x.data, window))):
            corr_data[i] = np.nan if i < window-1 else weighted_corr(np.array(piece1), np.array(piece2))
    else:
        assert half_life is not None, "Parameter half_life must be specified if weight is True."
        decay = time_weighted(window , half_life)

        for i, (piece1, piece2) in enumerate(zip(rolling(y.data, window), rolling(x.data, window))):
            corr_data[i] = np.nan if i < window-1 else weighted_corr(piece1, piece2, weights=decay)
    
    corr_data = pd.DataFrame(corr_data, index=y.data.index, columns=y.data.columns)
    corr_expr = f"Corr({y.expr}, {x.expr}, {window})"

    return Factor(corr_data, corr_expr)


def ts_cumsum_agg(x, window, agg_func="max", direction="forward"):
    
    arr = x.data.values
    cumsum_max, cumsum_min = numba_cumsum_extremes(arr, window, direction)

    if agg_func == "max":
        cumsum_data = pd.DataFrame(cumsum_max, index=x.data.index, columns=x.data.columns)
    elif agg_func == "min":
        cumsum_data = pd.DataFrame(cumsum_min, index=x.data.index, columns=x.data.columns)
    else: pass
    
    cumsum_expr = f"Cumsum({x.expr}, {window}, {agg_func})"

    return Factor(cumsum_data, cumsum_expr)


def apply(x, func, window=None, **kwargs):

    if window is not None:
        apply_data = x.data.rolling(window).apply(func, **kwargs)
        apply_expr = f"Apply({x.expr}, {func.__name__}, {window})"
    else:
        apply_data = x.data.apply(func, **kwargs)
        apply_expr = f"Apply({x.expr}, {func.__name__})"
    
    return Factor(apply_data, apply_expr)


def num_gt_mean(x, window):

    data = pd.DataFrame(numba_num_gt_mean(np.array(x.data), window), index=x.data.index, columns=x.data.columns)
    expr = f"n_gt_Mean({x.expr})"

    return Factor(data, expr)


def num_lt_mean(x, window):

    data = pd.DataFrame(numba_num_lt_mean(np.array(x.data), window), index=x.data.index, columns=x.data.columns)
    expr = f"n_lt_Mean({x.expr})"

    return Factor(data, expr)


def lower_center_squared_sum(x, window):

    data = pd.DataFrame(numba_lcenter_squared_sum(np.array(x.data), window), index=x.data.index, columns=x.data.columns)
    expr = f"l_center_Sq_Sum({x.expr})"

    return Factor(data, expr)


def upper_center_squared_sum(x, window):

    data = pd.DataFrame(numba_ucenter_squared_sum(np.array(x.data), window), index=x.data.index, columns=x.data.columns)
    expr = f"u_center_Sq_Sum({x.expr})"

    return Factor(data, expr)


def ts_linreg(y, x, window, weight=False, half_life=None, param="beta", compute_mode="auto"):
    """
    Rolling linear regression supporting single or multiple factors.
    
    Args:
        y: dependent variable (Factor)
        x: independent variable(s) - can be a single Factor or a list of Factors
        window: rolling window size
        weight: whether to use weighted regression
        half_life: half-life for exponential weights
        param: "beta", "alpha", or "resid"
        compute_mode: "standard", "vectorized", "numba", or "auto"
    
    Returns:
        Factor with appropriate structure based on param and number of x factors
    """
    
    n_times, n_tickers = y.data.shape
    time_index, ticker_columns = y.data.index, y.data.columns

    # Handle single vs multiple x factors
    if isinstance(x, list) and len(x) > 1:
        x_factors = x
        is_multi_factor = True
    else:
        x_factors = [x]
        is_multi_factor = False
    
    # Prepare weights
    weights = None
    if weight:
        assert half_life is not None, "`half_life` must be specified if `weight` is True."
        weights = time_weighted(window, half_life)
    
    # Get expressions for output
    x_exprs = [xf.expr for xf in x_factors]
    x_expr_str = f"({', '.join(x_exprs)})" if is_multi_factor else x_exprs[0]
    
    # Pre-allocate arrays for better performance
    y_array = y.data.values
    x_arrays = [xf.data.values for xf in x_factors]
    data_size = n_times * n_tickers
    
    # Compute mode selection
    if compute_mode == "auto":
        # Auto-select based on data size and availability
        if data_size > 10000 and has_numba():
            compute_mode = "numba"
        elif data_size > 25000:
            compute_mode = "vectorized"
        else:
            compute_mode = "standard"
    
    # Execute computation based on mode
    if compute_mode == "numba":
        if not has_numba():
            raise ImportError("Numba not available")
        return ts_linreg_numba(y, x_factors, y_array, x_arrays, window, weights, param, time_index, ticker_columns, x_expr_str, is_multi_factor)
    
    elif compute_mode == "vectorized":
        return ts_linreg_vectorized(y, x_factors, y_array, x_arrays, window, weights, param, time_index, ticker_columns, x_expr_str, is_multi_factor)
    
    elif compute_mode == "standard":
        return ts_linreg_standard(y, x_factors, y_array, x_arrays, window, weights, param, time_index, ticker_columns, x_expr_str, is_multi_factor)
    
    else:
        raise ValueError(f"Invalid compute_mode: {compute_mode}. Must be 'standard', 'vectorized', 'numba', or 'auto'")


def resample_agg(x, freq, agg_func="mean", min_periods=0, label="right", cal_rule="calendar"):
    """
    Resample time-series data by a specified frequency and apply an aggregation function.
    Note: The last group will be dropped if it doesn't have enough data points to complete a period.
    
    Parameters:
        freq: Resampling frequency. Can be either:
            - int: Represents the window size for fixed-window resampling.
            - str: Represents time-based frequency (e.g., 'D' for daily, 'W' for weekly)
                for time-based resampling.
        agg_func: Union[str, callable], optional
            Aggregation function to apply to each resampled group. Defaults to "mean".
            Common options include "sum", "max", "min", "std", etc.
            Supports user-customized functions.
        min_periods: int, optional
            Minimum number of non-NA values required in a group to compute the aggregation result.
            If not met, the group result will be NA.
        label: str, optional
            Label for the resampled groups. Defaults to "right".
            "right" means using the end of the interval as the label, "left" uses the start.
        cal_rule: str, optional
            Calendar rule for fixed-window resampling. Defaults to "calendar".
            Only used when `freq` is a fixed window (integer).
    
    Returns:
        Factor object with period (normally non-daily) frequency.
    """

    if isinstance(freq, int):
        resample_data = agg_by_window(x.data, freq, agg_func, min_periods, label, cal_rule)
    
    else:  # str
        resample_data = agg_by_resampler(x.data, freq, agg_func, min_periods, label)
    
    resample_expr = f"ResampleAgg({x.expr}, {freq}, {agg_func})"

    return Factor(resample_data, resample_expr)


def block_agg(x, window: int, sections: int, inner_func: object, outer_func: object):
    """
    Perform a two-level aggregation within rolling windows by splitting each window into blocks.
    Note: The window size should be fully divided by number of sections.
    
    Parameters:
        window: int
            Size of the rolling window (number of rows per window).
        sections: int
            Number of row-based blocks to split each rolling window into.
        inner_func: object
            Function to apply to each block within a window. Should accept a single
            block and return an aggregated value.
        outer_func: object
            Function to apply to the list of results from `inner_func`. Should accept
            an iterable of inner aggregation results and return a single value for the window.
    
    Returns:
        Factor object with daily frequency.
    
    Examples:
        >>> # Calculate range of means: split each 10-row window into 2 blocks, compute mean per block, then range of means
        >>> block_agg(data, window=10, sections=2, inner_func=lambda x: x.mean(), outer_func=lambda x: max(x)-min(x))
    """
    def _block_func(window_data):
        blocks = split_dataframe_by_rows(window_data, n_splits=sections)
        inner_agg_result = [inner_func(block) for block in blocks]
        outer_agg_result = outer_func(inner_agg_result)
        return outer_agg_result

    block_data = apply(x, _block_func, window).data
    block_expr = f"BlockAgg({x.expr}, {window}, {sections}, {outer_func}({inner_func}))"

    return Factor(block_data, block_expr)


#================================= CROSS-SECTIONAL OPS =================================


def cs_mean(x, weight=None):
    """
    Calculate cross-sectional mean excluding the current column for each element.
    Supports simple average or weighted average based on provided weights.
    """
    x_data = x.data.copy()
    _, n_cols = x_data.shape

    if n_cols <= 1:
        mean_data = pd.DataFrame(np.nan, index=x_data.index, columns=x_data.columns)
    else:
        if weight is None:
            row_sum = x_data.sum(axis=1).values.reshape(-1, 1)
            mean_data = (row_sum - x_data.values) / (n_cols - 1)
            mean_data = pd.DataFrame(mean_data, index=x_data.index, columns=x_data.columns)
        else:
            weight_data = weight.data
            row_weight_total = weight_data.sum(axis=1).values.reshape(-1, 1)
            row_weight_sum = row_weight_total - weight_data.values
            row_weight_sum = pd.DataFrame(row_weight_sum, index=x_data.index, columns=x_data.columns)

            weighted_total = (x_data * weight_data).sum(axis=1).values.reshape(-1, 1)
            weighted_sum = weighted_total - (x_data * weight_data).values
            weighted_sum = pd.DataFrame(weighted_sum, index=x_data.index, columns=x_data.columns)
            
            mean_data = weighted_sum / row_weight_sum.replace(0, np.nan)
    
    weight_expr = weight.expr if weight else "False"
    mean_expr = f"CS_Mean({x.expr}, weight={weight_expr})"
    
    return Factor(mean_data, mean_expr)


def cs_rank(x):

    ranked_data = x.data.rank(axis=1, ascending=False, method="dense")
    ranked_expr = f"Rank({x.expr})"

    return Factor(ranked_data, ranked_expr)


def cs_linreg(y, x, weight=None, param="beta"):
    """
    Cross-sectional linear regression
    Note that parameter x should be the independent variable
    """
    if y.data.shape != x.data.shape:
        raise ValueError("Input factors must have the same shape")
    
    cs_res = np.full(y.data.shape, np.nan)
    expr_map = {
        "beta": f"CS_Beta({y.expr}, {x.expr})",
        "alpha": f"CS_Alpha({y.expr}, {x.expr})",
        "resid": f"CS_Resid({y.expr}, {x.expr})"
    }
    
    for i in range(len(y.data)):
        y_vals = y.data.iloc[i].values
        x_vals = x.data.iloc[i].values
        
        valid_mask = ~np.isnan(y_vals) & ~np.isnan(x_vals)
        
        if weight is not None:
            w_vals = weight.data.iloc[i].values
            valid_mask &= ~np.isnan(w_vals)
            weights = w_vals[valid_mask]
        else:
            weights = None
        
        y_valid = y_vals[valid_mask]
        x_valid = x_vals[valid_mask]
        
        if len(y_valid) < 2:
            continue
        
        if weights is not None:
            X = np.column_stack([np.ones_like(x_valid), x_valid])
            W = np.diag(weights)
            XW = X.T @ W
            beta = np.linalg.inv(XW @ X) @ XW @ y_valid
        else:
            X = np.column_stack([np.ones_like(x_valid), x_valid])
            beta = np.linalg.inv(X.T @ X) @ X.T @ y_valid
        
        alpha = beta[0]
        coef = beta[1]
        
        resid = y_valid - (alpha + coef * x_valid)
        
        if param == "beta":
            result_value = coef
        elif param == "alpha":
            result_value = alpha
        elif param == "resid":
            result_vals = np.full(len(valid_mask), np.nan)
            result_vals[valid_mask] = resid
            cs_res[i] = result_vals
            continue
        else:
            raise ValueError(f"Invalid param: {param}")
        
        result_vals = np.full(len(valid_mask), np.nan)
        result_vals[valid_mask] = result_value
        cs_res[i] = result_vals
    
    cs_res = pd.DataFrame(cs_res, index=y.data.index, columns=y.data.columns)
    res_expr = expr_map[param]
    
    return Factor(cs_res, res_expr)


def cs_mean_by_industry(x, weight=None, industry=None, compute_mode="standard"):
    """
    Cross-sectional average by industry excluding self.
    """
    if weight is None:
        weight = copy(x, fill_value=1)
    
    # Create industry mapping dict
    industry_map = industry.data.set_index("ticker")["industry"].to_dict()
    
    # Map raw data columns to each industry while retaining the order
    industry_labels = x.data.columns.map(industry_map).values

    if compute_mode == "standard":
        cs_mean_data = cs_mean_by_industry_standard(x.data, weight.data, industry_labels)
    
    elif compute_mode == "vectorized":
        cs_mean_data = cs_mean_by_industry_vectorized(x.data, weight.data, industry_labels)
    
    else:
        raise ValueError(f"Invalid compute_mode: {compute_mode}. Must be one of 'standard', 'vectorized'")
    
    weight_expr = weight.expr if weight else "False"
    cs_mean_expr = f"CS_Mean_Ind({x.expr}, weight={weight_expr})"
    
    return Factor(cs_mean_data, cs_mean_expr)