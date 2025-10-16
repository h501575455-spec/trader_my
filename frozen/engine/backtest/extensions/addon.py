"""
Custom Backtrader analyzers for enhanced trading analysis.

This module contains custom analyzers for detailed trade analysis and key performance indicators
calculation specifically designed for quantitative trading strategies.
"""

import numpy as np
import pandas as pd
import backtrader as bt
from typing import Dict, List, Optional, Tuple, Union


class TradeListAnalyzer(bt.Analyzer):
    """
    Comprehensive trade list analyzer for closed trades.
    
    This analyzer records detailed information about each closed trade and provides
    comprehensive trade statistics including entry/exit prices, P&L, duration,
    and risk metrics like MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion).
    
    Returns:
        dict: Dictionary containing trade records with the following keys:
            - Ref: Reference number from Backtrader
            - Ticker: Instrument symbol/name
            - Direction: Trade direction ('long' or 'short')
            - Entry_Date: Entry date/time
            - Entry_Price: Entry price (weighted average for multiple entries)
            - Exit_Date: Exit date/time
            - Exit_Price: Exit price (weighted average for multiple exits)
            - Price_Return%: Price change percentage during trade
            - PnL: Absolute profit/loss
            - PnL%: Profit/loss percentage relative to broker value
            - Position_Size: Trade position size
            - Position_Value: Trade position value
            - Cumulative_PnL: Cumulative profit/loss up to this trade
            - Bar_Duration: Trade duration in price bars
            - PnL/Bar: Average profit/loss per bar
            - MFE%: Maximum Favorable Excursion as percentage of entry price
            - MAE%: Maximum Adverse Excursion as percentage of entry price
    """
    
    def __init__(self):
        """Initialize the trade list analyzer."""
        super(TradeListAnalyzer, self).__init__()
        self.trade_records = []
        self.cumulative_profit = 0.0

    def get_analysis(self) -> List[Dict]:
        """
        Return the list of trade records.
        
        Returns:
            List[Dict]: List of dictionaries containing trade information
        """
        return self.trade_records

    def notify_trade(self, trade) -> None:
        """
        Process trade notifications when trades are closed.
        
        Args:
            trade: Backtrader trade object containing trade information
        """
        if not trade.isclosed:
            return
            
        # Get current broker value for percentage calculations
        broker_value = self.strategy.broker.getvalue()
        
        # Determine trade direction
        first_event = trade.history[0].event
        direction = "long" if first_event.size > 0 else "short"
        
        # Extract entry and exit information
        entry_info = trade.history[0].status
        exit_info = trade.history[-1].status
        
        entry_price = entry_info.price
        exit_price = trade.history[-1].event.price
        entry_date = bt.num2date(entry_info.dt)
        exit_date = bt.num2date(exit_info.dt)
        
        # Convert to date objects for daily timeframes
        if trade.data._timeframe >= bt.TimeFrame.Days:
            entry_date = entry_date.date()
            exit_date = exit_date.date()

        # Calculate trade metrics
        if entry_price > 0:  # Avoid division by zero
            price_change_pct = (exit_price / entry_price - 1) * 100
        else:
            price_change_pct = 0.0
        
        pnl = exit_info.pnlcomm
        pnl_pct = (pnl / broker_value) * 100 if broker_value > 0 else 0.0
        duration_bars = exit_info.barlen
        pnl_per_bar = pnl / duration_bars if duration_bars > 0 else 0.0
        
        # Update cumulative profit
        self.cumulative_profit += pnl

        # Find maximum position size and value
        max_size = max_value = 0.0
        for record in trade.history:
            if abs(record.status.size) > abs(max_size):
                max_size = record.status.size
                max_value = record.status.value

        # Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        mfe_pct = mae_pct = 0.0
        try:
            price_series_length = duration_bars + 1
            highest_price = max(trade.data.high.get(ago=0, size=price_series_length))
            lowest_price = min(trade.data.low.get(ago=0, size=price_series_length))
            
            # Calculate percentage moves from entry price
            upside_move_pct = ((highest_price - entry_price) / entry_price) * 100
            downside_move_pct = ((lowest_price - entry_price) / entry_price) * 100
            
            if direction == "long":
                mfe_pct = upside_move_pct    # Favorable is up for long positions
                mae_pct = downside_move_pct  # Adverse is down for long positions
            else:  # short position
                mfe_pct = -downside_move_pct  # Favorable is down for short positions
                mae_pct = -upside_move_pct    # Adverse is up for short positions
                      
        except (ValueError, IndexError):
            # Handle cases where price data is insufficient
            mfe_pct = mae_pct = 0.0

        # Create trade record
        trade_record = {
            "Ref": trade.ref,
            "Ticker": trade.data._name,
            "Direction": direction,
            "Entry_Date": entry_date,
            "Entry_Price": round(entry_price, 2),
            "Exit_Date": exit_date,
            "Exit_Price": round(exit_price, 2),
            "Price_Return%": round(price_change_pct, 2),
            "PnL": round(pnl, 2),
            "PnL%": round(pnl_pct, 2),
            "Position_Size": max_size,
            "Position_Value": round(max_value, 2),
            "Cumulative_PnL": round(self.cumulative_profit, 2),
            "Bar_Duration": duration_bars,
            "PnL/Bar": round(pnl_per_bar, 2),
            "MFE%": round(mfe_pct, 2),
            "MAE%": round(mae_pct, 2)
        }
        
        self.trade_records.append(trade_record)


class KeyIndicatorAnalyzer(bt.Analyzer):
    """
    Comprehensive key performance indicators analyzer.
    
    This analyzer calculates essential trading performance metrics including returns,
    risk measures, and trading statistics. It provides both strategy and benchmark
    comparisons with detailed daily tracking capabilities.
    
    Key Metrics Calculated:
        - Total Return: Cumulative return over the entire period
        - Annualized Return: Yearly compounded return rate
        - Maximum Drawdown: Largest peak-to-trough decline
        - Win Rate: Percentage of profitable trades
        - Sharpe Ratio: Risk-adjusted return measure
        - Kelly Ratio: Optimal position sizing indicator
        - Recent Period Returns: Short-term performance (7 and 30 days)
        - Commission Ratio: Trading costs as percentage of assets
        - Trade Count: Total number of completed trades
    """
    
    # Class constants for period calculations
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5
    RISK_FREE_RATE_ANNUAL = 0.00  # 3% annual risk-free rate
    
    def __init__(self):
        """Initialize the key indicator analyzer."""
        super(KeyIndicatorAnalyzer, self).__init__()
        
        # Daily portfolio tracking
        self.daily_portfolio_data = []
        
        # Trading cost tracking
        self.total_commission = 0.0
        
        # Trade outcome tracking
        self.winning_trades = []  # List of profitable trade PnLs
        self.losing_trades = []   # List of losing trade PnLs
        
        # Results storage
        self.performance_indicators = pd.DataFrame(
            columns=[
                "Strategy_Name",
                "Total_Return",
                "Annualized_Return",
                "Max_Drawdown",
                "Win_Rate",
                "Sharpe_Ratio",
                "Kelly_Ratio",
                "Recent_7D_Return",
                "Recent_30D_Return",
                "Commission_Ratio",
                "Total_Trades"
            ]
        )
        
        # Daily performance data for charting
        self.daily_performance_data = {}

    def next(self) -> None:
        """
        Record daily portfolio metrics.
        
        Called on each trading day to track portfolio value and cash positions.
        """
        current_date = self.strategy.data.datetime.date(0)
        portfolio_value = self.strategy.broker.getvalue()
        available_cash = self.strategy.broker.getcash()
        
        self.daily_portfolio_data.append({
            "date": current_date,
            "portfolio_value": portfolio_value,
            "cash": available_cash
        })

    def notify_trade(self, trade) -> None:
        """
        Process completed trades for performance analysis.
        
        Args:
            trade: Backtrader trade object containing trade details
        """
        if not trade.isclosed:
            return
            
        # Track commission costs
        self.total_commission += trade.commission
        
        # Categorize trade outcomes
        trade_pnl = trade.pnlcomm
        if trade_pnl >= 0:
            self.winning_trades.append(trade_pnl)
        else:
            self.losing_trades.append(trade_pnl)

    def stop(self) -> None:
        """
        Calculate final performance indicators when strategy completes.
        
        This method is called at the end of the backtest to compute all
        performance metrics and prepare the final analysis results.
        """
        # Convert daily data to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.daily_portfolio_data)
        
        if portfolio_df.empty:
            return
            
        portfolio_values = portfolio_df["portfolio_value"]
        
        # Calculate core performance metrics
        strategy_metrics = self._calculate_strategy_metrics(portfolio_values)
        
        # Add strategy results to performance indicators
        self.performance_indicators.loc[len(self.performance_indicators)] = [
            "strategy",
            strategy_metrics["Total_Return"],
            strategy_metrics["Annualized_Return"],
            strategy_metrics["Max_Drawdown"],
            strategy_metrics["Win_Rate"],
            strategy_metrics["Sharpe_Ratio"],
            strategy_metrics["Kelly_Ratio"],
            strategy_metrics["Recent_7D_Return"],
            strategy_metrics["Recent_30D_Return"],
            strategy_metrics["Commission_Ratio"],
            strategy_metrics["Total_Trades"]
        ]
        
        # Prepare daily performance data for visualization
        portfolio_df["return_curve"] = self._calculate_return_curve(portfolio_values)
        portfolio_df.set_index("date", inplace=True)
        self.daily_performance_data["strategy"] = portfolio_df

    def get_analysis_data(self, benchmark_df: pd.DataFrame, 
                         benchmark_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Get comprehensive analysis including benchmark comparison.
        
        Args:
            benchmark_df: DataFrame containing benchmark price data with 'close' column
            benchmark_name: Name identifier for the benchmark
            
        Returns:
            Tuple containing:
                - Performance indicators DataFrame
                - Daily performance data dictionary
        """
        self._calculate_benchmark_metrics(benchmark_df, benchmark_name)
        return self.performance_indicators, self.daily_performance_data

    def _calculate_strategy_metrics(self, portfolio_values: pd.Series) -> Dict:
        """
        Calculate comprehensive strategy performance metrics.
        
        Args:
            portfolio_values: Series of daily portfolio values
            
        Returns:
            Dictionary containing calculated performance metrics
        """
        # Win rate calculation
        total_trades = len(self.winning_trades) + len(self.losing_trades)
        win_rate = (len(self.winning_trades) / total_trades * 100 
                   if total_trades > 0 else 0)
        
        return {
            "Total_Return": self._calculate_total_return(portfolio_values),
            "Annualized_Return": self._calculate_annualized_return(portfolio_values),
            "Max_Drawdown": self._calculate_max_drawdown(portfolio_values),
            "Win_Rate": f"{win_rate:.2f}%" if total_trades > 0 else "N/A",
            "Sharpe_Ratio": self._calculate_sharpe_ratio(portfolio_values),
            "Kelly_Ratio": self._calculate_kelly_ratio(),
            "Recent_7D_Return": self._calculate_recent_return(
                portfolio_values, self.TRADING_DAYS_PER_WEEK
            ),
            "Recent_30D_Return": self._calculate_recent_return(
                portfolio_values, self.TRADING_DAYS_PER_MONTH
            ),
            "Commission_Ratio": self._calculate_commission_ratio(portfolio_values),
            "Total_Trades": total_trades
        }

    def _calculate_benchmark_metrics(self, benchmark_df: pd.DataFrame, 
                                   benchmark_name: str) -> None:
        """
        Calculate benchmark performance metrics for comparison.
        
        Args:
            benchmark_df: DataFrame with benchmark price data
            benchmark_name: Identifier for the benchmark
        """
        if "close" not in benchmark_df.columns:
            raise ValueError("Benchmark DataFrame must contain 'close' column")
            
        benchmark_prices = benchmark_df["close"]
        
        # Calculate benchmark metrics (no trading-specific metrics)
        benchmark_metrics = [
            benchmark_name,
            self._calculate_total_return(benchmark_prices),
            self._calculate_annualized_return(benchmark_prices),
            self._calculate_max_drawdown(benchmark_prices),
            None,  # No win rate for benchmark
            self._calculate_sharpe_ratio(benchmark_prices),
            None,  # No Kelly ratio for benchmark
            self._calculate_recent_return(benchmark_prices, self.TRADING_DAYS_PER_WEEK),
            self._calculate_recent_return(benchmark_prices, self.TRADING_DAYS_PER_MONTH),
            None,  # No commission for benchmark
            None   # No trades for benchmark
        ]
        
        self.performance_indicators.loc[len(self.performance_indicators)] = benchmark_metrics
        
        # Prepare benchmark daily data
        benchmark_daily = pd.DataFrame(index=benchmark_df.index)
        benchmark_daily["return_curve"] = self._calculate_return_curve(benchmark_prices)
        benchmark_daily.index.name = "date"
        self.daily_performance_data[benchmark_name] = benchmark_daily

    def _calculate_total_return(self, price_series: pd.Series) -> str:
        """Calculate total return over the entire period."""
        if len(price_series) < 2:
            return "0.00%"
        total_return = (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]
        return f"{total_return * 100:.2f}%"

    def _calculate_annualized_return(self, price_series: pd.Series) -> str:
        """Calculate annualized return."""
        if len(price_series) < 2:
            return "0.00%"
        total_return = (price_series.iloc[-1] - price_series.iloc[0]) / price_series.iloc[0]
        annualized_return = total_return / len(price_series) * self.TRADING_DAYS_PER_YEAR
        return f"{annualized_return * 100:.2f}%"

    def _calculate_recent_return(self, price_series: pd.Series, period_days: int) -> str:
        """Calculate return for recent period."""
        if len(price_series) < period_days + 1:
            return "0.00%"
        recent_return = (price_series.iloc[-1] - price_series.iloc[-period_days-1]) / price_series.iloc[-period_days-1]
        return f"{recent_return * 100:.2f}%"

    def _calculate_max_drawdown(self, price_series: pd.Series) -> str:
        """Calculate maximum drawdown."""
        if len(price_series) < 2:
            return "0.00%"
        running_max = price_series.expanding().max()
        drawdowns = (price_series - running_max) / running_max
        max_drawdown = drawdowns.min()
        return f"{max_drawdown * 100:.2f}%"

    def _calculate_sharpe_ratio(self, price_series: pd.Series) -> float:
        """
        Calculate Sharpe ratio.
        
        The Sharpe ratio measures risk-adjusted returns by comparing the excess return
        (above risk-free rate) to the standard deviation of returns.
        
        Formula: (Average Return - Risk-Free Rate) / Standard Deviation of Returns
        """
        if len(price_series) < 2:
            return 0.0
            
        # Calculate daily returns
        daily_returns = price_series.pct_change().fillna(0)
        
        if daily_returns.std() == 0:
            return 0.0
            
        # Calculate average daily return and risk-free rate
        avg_daily_return = daily_returns.mean()
        daily_risk_free_rate = self.RISK_FREE_RATE_ANNUAL / self.TRADING_DAYS_PER_YEAR
        
        # Calculate daily Sharpe ratio and annualize it
        daily_sharpe = (avg_daily_return - daily_risk_free_rate) / daily_returns.std()
        annualized_sharpe = daily_sharpe * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        
        return round(annualized_sharpe, 3)

    def _calculate_kelly_ratio(self) -> Optional[str]:
        """
        Calculate Kelly criterion for optimal position sizing.
        
        The Kelly criterion determines the optimal fraction of capital to risk
        on each trade to maximize long-term growth.
        
        Formula: K = W - [(1 - W) / R]
        Where:
            - W = Win rate (probability of winning)
            - R = Win/Loss ratio (average win / average loss)
            - K = Kelly percentage
        
        Returns:
            Optimal position sizing percentage or None if insufficient data
        """
        win_count = len(self.winning_trades)
        loss_count = len(self.losing_trades)
        
        if win_count == 0 or loss_count == 0:
            return None
            
        # Calculate win rate and average win/loss amounts
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades
        avg_win = np.mean(self.winning_trades)
        avg_loss = abs(np.mean(self.losing_trades))  # Take absolute value
        
        if avg_loss == 0:
            return None
            
        # Calculate win/loss ratio and Kelly percentage
        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        return f"{kelly_percentage * 100:.2f}%"

    def _calculate_commission_ratio(self, portfolio_values: pd.Series) -> str:
        """Calculate commission costs as percentage of initial capital."""
        if len(portfolio_values) == 0 or portfolio_values.iloc[0] == 0:
            return "0.00%"
        commission_ratio = self.total_commission / portfolio_values.iloc[0]
        return f"{commission_ratio * 100:.2f}%"

    def _calculate_return_curve(self, price_series: pd.Series) -> pd.Series:
        """Calculate cumulative return curve as percentage."""
        if len(price_series) == 0:
            return pd.Series(dtype=float)
        return_curve = (price_series - price_series.iloc[0]) / price_series.iloc[0] * 100
        return return_curve.round(2)

    def _get_winning_trade_count(self) -> int:
        """Get number of winning trades."""
        return len(self.winning_trades)

    def _get_losing_trade_count(self) -> int:
        """Get number of losing trades."""
        return len(self.losing_trades)