# Preprocessing Operators

import numpy as np
import pandas as pd
from typing import Union, Tuple
from scipy.stats import norm
from functools import partial
from scipy.stats.mstats import winsorize

from ..expression.base import Factor

# Try to import sklearn for regression, fallback to numpy if not available
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def normalize(
    factor: Union[Factor, pd.DataFrame],
    cross_section: bool = False,
    **kwargs
) -> Union[Factor, pd.DataFrame]:
    """
    Normalize factor data using min-max scaling.
    For time-series normalization, rolling window or expanding is applied to avoid future data.
    For cross-sectional normalization, the standard approach is used.

    Parameters:
    -----------
    cross_section: bool
        True if cross-sectional data
        False if time-series data (default)
    kwargs:
        - expanding: bool, optional 
            True if include all previous data
            False if include only rolling period data (default)
        - window: int, optional
            Rolling window
    """
    
    # Extract data based on input type
    if isinstance(factor, Factor):
        data = factor.data.copy()
        expr = factor.expr
        multiindex_flag = False
    elif isinstance(factor, pd.DataFrame):
        if isinstance(factor.index, pd.MultiIndex):
            data = factor
            multiindex_flag = True
        else:
            raise ValueError(f"Incorrect factor index type, should be pd.MultiIndex, got {type(factor.index)} instead.")
    else:
        raise ValueError("Incorrect factor input type")
    
    if multiindex_flag:
        if cross_section:
            # group by date
            group = data.groupby("trade_date", group_keys=False)
            min_val = group.min()
            max_val = group.max()
        else:
            # Validate parameter
            expanding = kwargs.get("expanding")
            window = kwargs.get("window")
            if expanding is None:
                raise ValueError("`expanding` should not be none if `cross_section` is False")
            if expanding and window is not None:
                raise ValueError("expanding method takes no `window` parameter")
            if not expanding and window is None:
                raise ValueError("`window` should not be none if `expanding` is False")
            
            # group by ticker
            group = data.groupby("ticker", group_keys=False)
            if expanding:
                min_val = group.expanding(min_periods=1).min().droplevel(0)
                max_val = group.expanding(min_periods=1).max().droplevel(0)
            else:
                min_val = group.rolling(window, min_periods=1).min().droplevel(0)
                max_val = group.rolling(window, min_periods=1).max().droplevel(0)
        
        norm_data = (data - min_val) / (max_val - min_val)
        
        return norm_data
    
    else:
        if cross_section:
            min_val = data.min(axis=1)
            max_val = data.max(axis=1)
            norm_data = data.sub(min_val, axis=0).div(max_val - min_val, axis=0)
        else:
            # Validate parameter
            expanding = kwargs.get("expanding")
            window = kwargs.get("window")
            if expanding is None:
                raise ValueError("`expanding` should not be none if `cross_section` is False")
            if expanding and window is not None:
                raise ValueError("expanding method takes no `window` parameter")
            if not expanding and window is None:
                raise ValueError("`window` should not be none if `expanding` is False")
            
            if expanding:
                min_val = data.expanding(min_periods=1).min()
                max_val = data.expanding(min_periods=1).max()
            else:
                min_val = data.rolling(window, min_periods=1).min()
                max_val = data.rolling(window, min_periods=1).max()
            
            norm_data = (data - min_val) / (max_val - min_val)
        
        norm_expr = f"Normalize({expr})"
        return Factor(norm_data, norm_expr)


def standardize(
    factor: Union[Factor, pd.DataFrame],
    cross_section: bool = False,
    **kwargs
) -> Union[Factor, pd.DataFrame]:
    """
    Standardize factor data by computing z-score.
    For time-series standardization, rolling window or expanding is applied to avoid future data.
    For cross-sectional standardization, standard approach is used.

    Parameters:
    -----------
    cross_section: bool
        True if cross-sectional data
        False if time-series data (default)
    kwargs:
        - expanding: bool, optional 
            True if include all previous data
            False if include only rolling period data (default)
        - window: int, optional
            Rolling window
    """

    # Extract data based on input type
    if isinstance(factor, Factor):
        data = factor.data.copy()
        expr = factor.expr
        multiindex_flag = False
    elif isinstance(factor, pd.DataFrame):
        if isinstance(factor.index, pd.MultiIndex):
            data = factor
            multiindex_flag = True
        else:
            raise ValueError(f"Incorrect factor index type, should be pd.MultiIndex, got {type(factor.index)} instead.")
    else:
        raise ValueError("Incorrect factor input type")

    if multiindex_flag:
        if cross_section:
            # group by date
            group = data.groupby("trade_date", group_keys=False)
            mean_val = group.mean()
            std_val = group.std()
        else:
            # Validate parameter
            expanding = kwargs.get("expanding")
            window = kwargs.get("window")
            if expanding is None:
                raise ValueError("`expanding` should not be none if `cross_section` is False")
            if expanding and window is not None:
                raise ValueError("expanding method takes no `window` parameter")
            if not expanding and window is None:
                raise ValueError("`window` should not be none if `expanding` is False")
            
            # group by ticker
            group = data.groupby("ticker", group_keys=False)
            if expanding:
                mean_val = group.expanding(min_periods=1).mean().droplevel(0)
                std_val = group.expanding(min_periods=1).std().droplevel(0)
            else:
                mean_val = group.rolling(window, min_periods=1).mean().droplevel(0)
                std_val = group.rolling(window, min_periods=1).std().droplevel(0)
        
        standard_data = (data - mean_val) / std_val
        
        return standard_data
    
    else:
        if cross_section:
            mean_val = data.mean(axis=1)
            std_val = data.std(axis=1)
            standard_data = data.sub(mean_val, axis=0).div(std_val, axis=0)
        else:
            # Validate parameter
            expanding = kwargs.get("expanding")
            window = kwargs.get("window")
            if expanding is None:
                raise ValueError("`expanding` should not be none if `cross_section` is False")
            if expanding and window is not None:
                raise ValueError("expanding method takes no `window` parameter")
            if not expanding and window is None:
                raise ValueError("`window` should not be none if `expanding` is False")
            
            if expanding:
                mean_val = data.expanding(min_periods=1).mean()
                std_val = data.expanding(min_periods=1).std()
            else:
                mean_val = data.rolling(window, min_periods=1).mean()
                std_val = data.rolling(window, min_periods=1).std()
            
            standard_data = (data - mean_val) / std_val
        
        standard_expr = f"Standardize({expr})"
        return Factor(standard_data, standard_expr)


def clip_mad(
    factor: Factor,
    multiplier: float = 3.0,
    cross_section: bool = False,
    **kwargs
) -> Factor:
    """
    Clip factor data using MAD (Median Absolute Deviation) method.
    
    Parameters:
    -----------
    factor: Factor
        Factor object to be clipped
    multiplier: float
        MAD multiplier for outlier detection (default: 3.0)
    cross_section: bool
        True if cross-sectional data
        False if time-series data (default)
    kwargs:
        - expanding: bool, optional 
            True if include all previous data
            False if include only rolling period data (default)
        - window: int, optional
            Rolling window size for time-series mode
        
    Returns:
    --------
    Factor: Clipped factor object
    """

    data = factor.data.copy()
    expr = factor.expr

    if cross_section:
        clip_data = np.full(data.shape, np.nan)
        
        for i in range(len(data)):
            cross_section = data.iloc[i].values
            
            if np.all(np.isnan(cross_section)):
                continue
            
            med = np.nanmedian(cross_section)
            abs_dev = np.abs(cross_section - med)
            mad = np.nanmedian(abs_dev)
            
            if mad < 1e-6:
                clip_data[i] = cross_section
                continue
                
            scaled_mad = 1.4826 * mad
            lower_bound = med - multiplier * scaled_mad
            upper_bound = med + multiplier * scaled_mad
            
            clipped_section = np.clip(cross_section, lower_bound, upper_bound)
            clip_data[i] = clipped_section
        
        clip_data = pd.DataFrame(clip_data, index=data.index, columns=data.columns)
    
    else:
        expanding = kwargs.get("expanding")
        window = kwargs.get("window")

        if expanding is None:
            raise ValueError("`expanding` should not be none if `cross_section` is False")
        if expanding and window is not None:
            raise ValueError("expanding method takes no `window` parameter")
        if not expanding and window is None:
            raise ValueError("`window` should not be none if `expanding` is False")
        
        if expanding:
            assert window is None, "expanding method takes no parameter as `window`."
            med = data.expanding(min_periods=1).median()
            mad = data.expanding(min_periods=1).apply(lambda x: abs(x - x.median()).median())
        
        else:
            med = data.rolling(window, min_periods=1).median()
            mad = data.rolling(window, min_periods=1).apply(lambda x: abs(x - x.median()).median())
        
        upper_bound = med + (multiplier * 1.4826 * mad)
        lower_bound = med - (multiplier * 1.4826 * mad)

        clip_data = data.clip(lower=lower_bound, upper=upper_bound)
    
    clip_expr = f"ClipMAD({expr})"

    return Factor(clip_data, clip_expr)


def winsorize_quantile(
    factor: Factor,
    limits: Tuple = (0.1, 0.1),
    cross_section: bool = False,
    **kwargs
) -> Factor:
    """
    Winsorize factor data using quantile method.
    
    Parameters:
    -----------
    factor: Factor
        Factor object to be winsorized
    limits: Tuple
        Tuple of lower and upper quantile limits (default: (0.1, 0.1))
    cross_section: bool
        True for cross-sectional winsorization (each time point independently)
        False for time-series winsorization (default)
    kwargs:
        - expanding: bool, optional 
            True if include all previous data
            False if include only rolling period data (default)
        - window: int, optional
            Rolling window size for time-series mode
        
    Returns:
    --------
    Factor: Winsorized factor object
    """

    data = factor.data.copy()
    expr = factor.expr

    if cross_section:
        win_data = data.copy()
        for i in range(len(data)):
            cross_section = data.iloc[i].values

            if np.all(np.isnan(cross_section)):
                continue
                
            try:
                winsorized = winsorize(cross_section, limits=limits, nan_policy='omit')
                win_data.iloc[i] = winsorized
            except ValueError:
                win_data.iloc[i] = cross_section
    else:
        expanding = kwargs.get("expanding")
        window = kwargs.get("window")

        if expanding is None:
            raise ValueError("`expanding` should not be none if `cross_section` is False")
        if expanding and window is not None:
            raise ValueError("expanding method takes no `window` parameter")
        if not expanding and window is None:
            raise ValueError("`window` should not be none if `expanding` is False")

        if expanding:
            assert window is None, "expanding method takes no parameter as `window`."
            lower_limit = data.expanding(min_periods=1).apply(lambda x: winsorize(x, limits=limits, nan_policy="omit").min(), raw=False)
            upper_limit = data.expanding(min_periods=1).apply(lambda x: winsorize(x, limits=limits, nan_policy="omit").max(), raw=False)

        else:
            lower_limit = data.rolling(window, min_periods=1).apply(lambda x: winsorize(x, limits=limits, nan_policy="omit").min(), raw=False)
            upper_limit = data.rolling(window, min_periods=1).apply(lambda x: winsorize(x, limits=limits, nan_policy="omit").max(), raw=False)
        
        win_data = data.clip(lower=lower_limit, upper=upper_limit)
    
    win_expr = f"WinsorizeQuantile({expr})"
    return Factor(win_data, win_expr)


def industry_neutralize(
        factor: Union[Factor, pd.DataFrame],
        industry_mapping,
        method="demean",
        multiindex=False
    ):
    """
    Neutralize factor by industry mapping.
    
    Parameters:
    -----------
    factor : Factor or pd.DataFrame
        Factor data. If multiindex=False, should be Factor object.
        If multiindex=True, should be DataFrame with MultiIndex (ticker, trade_date) or long format.
    industry_mapping : pd.DataFrame
        Industry mapping DataFrame. Format depends on multiindex parameter:
        - If multiindex=False: index is date, column is ticker, value is industry
        - If multiindex=True: DataFrame with columns ['ticker', 'industry']
    method : str
        Neutralization method:
        - 'demean': Industry-wise mean subtraction
        - 'standardize': Industry-wise standardization (mean + standardization)
        - 'rank': Industry-wise ranking and standardization
        - 'regression': Linear regression against industry dummy variables, using residuals
    multiindex : bool
        Data format flag:
        - False: Use Factor object format (default)
        - True: Use pandas MultiIndex format
    
    Returns:
    --------
    Factor or pd.DataFrame: Industry-neutralized factor object/data
    """
    
    if not multiindex:
        # Traditional Factor object format
        if not isinstance(factor, Factor):
            raise ValueError("When multiindex=False, factor must be a Factor object")
        
        if not isinstance(industry_mapping.index, pd.DatetimeIndex):
            raise ValueError("Industry mapping index must be a DatetimeIndex")
        
        factor_data = factor.data.copy()
        factor_expr = factor.expr

        # Ensure data index and columns alignment
        aligned_industry = industry_mapping.reindex(
            index=factor_data.index, 
            columns=factor_data.columns, 
            method="bfill"
        ).ffill()
        
        # Industry neutralization
        neutralized_data = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
        
        for date in factor_data.index:
            date_factor = factor_data.loc[date]
            date_industry = aligned_industry.loc[date]
            
            # Filter out nan values
            valid_mask = ~(pd.isna(date_factor) | pd.isna(date_industry))
            if not valid_mask.any():
                neutralized_data.loc[date] = date_factor
                continue
            
            valid_factor = date_factor[valid_mask]
            valid_industry = date_industry[valid_mask]
            
            if method == "demean":
                # Industry-wise mean
                industry_mean = valid_factor.groupby(valid_industry).transform("mean")
                neutralized_values = valid_factor - industry_mean
            
            elif method == "standardize":
                # Industry-wise standardization
                industry_mean = valid_factor.groupby(valid_industry).transform("mean")
                industry_std = valid_factor.groupby(valid_industry).transform("std")
                # Avoid division by zero
                industry_std = industry_std.fillna(1.0)
                industry_std = industry_std.replace(0.0, 1.0)
                neutralized_values = (valid_factor - industry_mean) / industry_std
            
            elif method == "rank":
                # Industry-wise ranking and standardization
                def rank_to_normal(group):
                    """Convert group values to standard normal distribution via ranking"""
                    ranks = group.rank(pct=True)
                    # Clip to avoid 0 and 1 which would result in infinity
                    ranks_clipped = ranks.clip(1e-6, 1-1e-6)
                    return norm.ppf(ranks_clipped)
                
                neutralized_values = valid_factor.groupby(valid_industry).transform(rank_to_normal)
            
            elif method == "regression":
                # Linear regression against industry dummy variables
                neutralized_values = _regression_neutralize(valid_factor, valid_industry)
            
            else:
                raise ValueError(f"Unsupported neutralization method: {method}")
            
            # Put processed values back to original position
            neutralized_data.loc[date, valid_mask] = neutralized_values
            # Keep original nan values
            neutralized_data.loc[date, ~valid_mask] = date_factor[~valid_mask]
        
        neutral_expr = f"IndustryNeutral({factor_expr})"
        return Factor(neutralized_data, neutral_expr)
    
    else:
        # MultiIndex format
        if not isinstance(factor, pd.DataFrame) or not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("When multiindex=True, factor must be a pandas DataFrame with MultiIndex")
        
        if not isinstance(industry_mapping.index, pd.MultiIndex):
            raise ValueError("Industry mapping index must be a MultiIndex(ticker, trade_date)")
        
        # Merge industry information
        merged_data = factor.merge(industry_mapping, left_index=True, right_index=True, how="left")
        
        # Apply neutralization by date and industry
        if method == "demean":
            # Industry-wise mean
            industry_mean = merged_data.groupby(["trade_date", "industry"]).transform("mean")
            neutralized_data = factor - industry_mean
        
        elif method == "standardize":
            # Industry-wise standardization
            def industry_standardize(group):
                mean_val = group.mean()
                std_val = group.std()
                if pd.isna(std_val) or std_val == 0:
                    return group - mean_val
                return (group - mean_val) / std_val
            
            neutralized_data = merged_data.groupby(["trade_date", "industry"]).transform(industry_standardize)
        
        elif method == "rank":
            # Industry-wise ranking and standardization
            def industry_rank_standardize(group):
                ranks = group.rank(pct=True)
                ranks_clipped = ranks.clip(1e-6, 1-1e-6)
                return norm.ppf(ranks_clipped)
            
            neutralized_data = merged_data.groupby(["trade_date", "industry"]).transform(industry_rank_standardize)
        
        elif method == "regression":
            # Linear regression against industry dummy variables by date
            def regression_by_date(group):
                factor_values = group.iloc[:, 0]  # First column is factor values
                industries = group["industry"]
                
                # Filter out NaN values
                valid_mask = ~(pd.isna(factor_values) | pd.isna(industries))
                if not valid_mask.any():
                    return factor_values  # Return original values if no valid data
                
                valid_factor = factor_values[valid_mask]
                valid_industry = industries[valid_mask]
                
                # Perform regression on valid data
                neutralized_values = _regression_neutralize(valid_factor, valid_industry)
                
                # Create result series with same index as original
                result = factor_values.copy()
                result.loc[valid_mask] = neutralized_values
                # Keep original NaN values unchanged
                result.loc[~valid_mask] = factor_values[~valid_mask]
                
                return result
            
            # Apply regression by date
            neutralized_data = merged_data.groupby("trade_date", group_keys=False).apply(regression_by_date)
        
        else:
            raise ValueError(f"Unsupported neutralization method: {method}")
        
        # Ensure proper index alignment for all methods
        neutralized_data = neutralized_data.reindex(factor.index)
        return neutralized_data


def _regression_neutralize(factor_values: pd.Series, industries: pd.Series) -> pd.Series:
    """
    Helper function to perform regression-based industry neutralization.
    
    Parameters:
    -----------
    factor_values : pd.Series
        Factor values for regression
    industries : pd.Series  
        Industry labels for each observation
        
    Returns:
    --------
    pd.Series: Regression residuals (neutralized factor values)
    """
    # Create industry dummy variables
    unique_industries = industries.unique()
    n_industries = len(unique_industries)
    
    # If only one industry, return original values (no regression needed)
    if n_industries <= 1:
        return factor_values
    
    # Create dummy variables matrix (exclude one industry to avoid multicollinearity)
    dummy_matrix = pd.get_dummies(industries, drop_first=True)
    
    # Perform regression
    if SKLEARN_AVAILABLE:
        # Use sklearn if available
        reg = LinearRegression(fit_intercept=True)
        reg.fit(dummy_matrix, factor_values)
        residuals = factor_values - reg.predict(dummy_matrix)
    else:
        # Use numpy least squares as fallback
        X = dummy_matrix.values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept
        y = factor_values.values
        
        try:
            # Solve normal equation: beta = (X'X)^(-1)X'y
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            predicted = X @ beta
            residuals = y - predicted
            residuals = pd.Series(residuals, index=factor_values.index)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Regression failed: {e}")
    
    return residuals


__all__ = ["normalize", "standardize", "clip_mad", "winsorize_quantile", "industry_neutralize"]
