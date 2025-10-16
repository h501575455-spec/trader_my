# helper functions for backtest framework

import os
import numpy as np
import pandas as pd
from typing import Union

def _lazy_import_factor():
    from ..factor.expression.base import Factor
    return Factor

def plot_fig(fig, write_html=False, path=""):
    # Save and display
    if write_html:
        if not path:
            path = os.getcwd()  # default
        else:
            if path.endswith("py"):
                path = os.path.join(os.path.dirname(path), "pages")
        os.makedirs(path, exist_ok=True)
    
    from plotly.graph_objs._figure import Figure
    from pyecharts.charts import Grid
    
    if isinstance(fig, Figure):

        file_name = f"{'_'.join(fig.layout.title.text.lower().split(' '))}.html"
        fig.write_html(os.path.join(path, file_name))
        print(f"Page saved as {file_name}")

        # Show in browser (if available)
        try:
            fig.show()
        except Exception as e:
            print(f"Cannot render in browser{e}")
    
    elif isinstance(fig, Grid):
        html_content = fig.render_embed()
        # Get theme background color, if not configured, use default value
        if hasattr(fig, 'config') and hasattr(fig.config, 'colors'):
            background_color = fig.config.colors.background_color
        else:
            background_color = "#fff"  # default white background
        
        centered_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>K线图表</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background-color: {background_color};
                    font-family: Arial, sans-serif;
                }}
                .chart-container {{
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    border-radius: 8px;
                    overflow: hidden;
                    background: {background_color};
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                {html_content}
            </div>
        </body>
        </html>
        """
        file_name = "pyechart_kline.html"
        # write to file
        with open(os.path.join(path, file_name), 'w', encoding='utf-8') as f:
            f.write(centered_html)

    else:
        raise ValueError(f"Unsupported figure type: {type(fig)}")
    
    return fig

def unpack_dict_for_init(original_dict):
    unpacked_dict = {}
    stack = [((), original_dict)]
    while stack:
        keys, current_dict = stack.pop()
        for k, v in current_dict.items():
            new_keys = keys + (k,)
            if isinstance(v, dict):
                stack.append((new_keys, v))
            else:
                unpacked_dict[k] = v
    return unpacked_dict

def label_to_string(label: str,
                    value: Union[float, str],
                    separator: str = "\n",
                    list_format: bool = False):
    """ Format label/value pairs for a unified formatting. """
    # Format option for lists such that all values are aligned:
    # Label: value1
    #        value2
    #        ...
    label = str(label)

    if list_format and isinstance(value, list) and len(value) > 0:
        s = label + ": "
        label_spacing = " " * len(s)
        s += str(value[0])

        for v in value[1:]:
            s += "\n" + label_spacing + str(v)
        s += separator

        return s

    return f"{label}: {value}{separator}"

def create_dummy_test_data(start_date="2023-01-01", end_date="2023-12-31", method="simple"):
    """
    Generate synthetic stock price and factor test data for demonstration.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    method: str
        - 'simple': generate simple ohlcv and single factor
        - 'advanced': generate ohlcv with some dynamics and multiple factors
        
    Returns:
    --------
    tuple: (price_data, factor_data)
    """
    # Generate synthetic data
    dates = pd.date_range(start_date, end_date, freq="B")
    n_days = len(dates)

    # Set random seed for reproducible results
    np.random.seed(42)
    
    if method == "simple":
        # Synthetic price data
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            "open": [p * np.random.uniform(0.99, 1.01) for p in prices],
            "high": [p * np.random.uniform(1.00, 1.05) for p in prices],
            "low": [p * np.random.uniform(0.95, 1.00) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        
        # Ensure OHLC relationships are correct
        for i in range(len(price_data)):
            high = max(price_data.iloc[i]["open"], price_data.iloc[i]["close"], price_data.iloc[i]["high"])
            low = min(price_data.iloc[i]["open"], price_data.iloc[i]["close"], price_data.iloc[i]["low"])
            price_data.iloc[i, price_data.columns.get_loc("high")] = high
            price_data.iloc[i, price_data.columns.get_loc("low")] = low
        
        # Synthetic factor data (somewhat correlated with future returns)
        factor_values = []
        for i in range(n_days):
            if i < n_days - 5:
                future_return = (prices[i+5] - prices[i]) / prices[i]
                factor = future_return + np.random.normal(0, 0.01)
            else:
                factor = np.random.normal(0, 0.02)
            factor_values.append(factor)
        
        factor_data = pd.Series(factor_values, index=dates, name='Dummy_Factor')
    
    elif method == "advanced":
        # Generate price data with trend and volatility
        base_price = 10.0  # Starting price
        trend = 0.0002  # Daily trend
        volatility = 0.025  # Daily volatility
        
        # Generate correlated returns
        returns = np.random.normal(trend, volatility, n_days)
        
        # Add some momentum/mean reversion patterns
        for i in range(1, n_days):
            if i >= 5:
                # Add momentum effect
                momentum = np.mean(returns[i-5:i]) * 0.1
                returns[i] += momentum
            
            # Add some mean reversion
            if returns[i-1] > 0.05:  # Large positive return
                returns[i] -= 0.01
            elif returns[i-1] < -0.05:  # Large negative return
                returns[i] += 0.01
        
        # Calculate prices
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove the initial price
        
        # Generate OHLC data
        price_data = []
        for i, close_price in enumerate(prices):
            # Generate intraday range
            daily_range = abs(np.random.normal(0, volatility * 0.5))
            
            # Open price (influenced by previous close)
            if i == 0:
                open_price = close_price * np.random.uniform(0.99, 1.01)
            else:
                gap = np.random.normal(0, volatility * 0.3)
                open_price = prices[i-1] * (1 + gap)
            
            # High and low
            high_price = max(open_price, close_price) * (1 + daily_range * np.random.uniform(0.2, 0.8))
            low_price = min(open_price, close_price) * (1 - daily_range * np.random.uniform(0.2, 0.8))
            
            # Volume (higher volume on bigger price moves)
            price_change = abs(returns[i]) if i < len(returns) else 0.01
            base_volume = 1000000
            volume = int(base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0))
            
            price_data.append({
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        price_data = pd.DataFrame(price_data, index=dates)
        
        # Generate factor data (simulating various factor types)
        factor_data = {}
        
        # 1. Momentum Factor (predicts future returns)
        momentum_values = []
        for i in range(n_days):
            if i >= 10:
                # 10-day price momentum
                past_return = (prices[i] - prices[i-10]) / prices[i-10]
                # Add noise and some predictive power
                factor_val = past_return * 0.7 + np.random.normal(0, 0.02)
            else:
                factor_val = np.random.normal(0, 0.02)
            momentum_values.append(factor_val)
        
        factor_data["momentum"] = pd.Series(momentum_values, index=dates, name="Momentum Factor")
        
        # 2. Volatility Factor
        vol_values = []
        for i in range(n_days):
            if i >= 20:
                # 20-day realized volatility
                recent_returns = returns[max(0, i-20):i]
                vol = np.std(recent_returns) * np.sqrt(252)  # Annualized volatility
                # Normalize to factor-like scale
                factor_val = (vol - 0.25) / 0.1  # Center around 0
            else:
                factor_val = np.random.normal(0, 1)
            vol_values.append(factor_val)
        
        factor_data["volatility"] = pd.Series(vol_values, index=dates, name="Volatility Factor")
        
        # 3. Mean Reversion Factor
        mean_rev_values = []
        for i in range(n_days):
            if i >= 5:
                # 5-day mean reversion
                recent_ret = np.mean(returns[max(0, i-5):i])
                # Mean reversion signal (negative of recent performance)
                factor_val = -recent_ret * 2 + np.random.normal(0, 0.01)
            else:
                factor_val = np.random.normal(0, 0.01)
            mean_rev_values.append(factor_val)
        
        factor_data["mean_reversion"] = pd.Series(mean_rev_values, index=dates, name="Mean Reversion Factor")
        
        factor_data = pd.DataFrame(factor_data)
    
    else:
        raise ValueError("Legitimate methods are 'simple' and 'advanced'")
    
    return price_data, factor_data
