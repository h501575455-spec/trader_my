import pandas as pd
import backtrader as bt
import vectorbt as vbt
from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod

from .StockEngine import FrozenBt
from ...utils.log import FrozenLogger
from ...utils.plotting import perf_plot


class PandasDataWithFactor(bt.feeds.PandasData):
    """Custom Pandas data feed that includes factor data as additional line."""
    lines = ("factor",)
    params = dict(
        factor=-1,  # Column index for factor data
    )


class RebalanceTableStrategy(bt.Strategy):
    """Multi-factor stock selection strategy based on rebalancing table.
    
    This strategy reads pre-calculated rebalancing decisions from a table
    and executes trades accordingly on scheduled rebalance dates.
    """
    
    params = dict(
        rebalance_data=None,  # DataFrame with columns: trade_date, instrument, weight
        cash_reserve=0.05,    # Reserve cash percentage to avoid insufficient funds
        trade_unit=100,       # Minimum order number for each trade
        log_trades=True,      # Whether to log trade executions
        sell_rule="current-bar",  # Trading timing rule
    )
    
    def __init__(self):
        """Initialize the rebalance table strategy."""
        if self.p.rebalance_data is None:
            raise ValueError("Rebalance data must be provided")
        
        # Extract unique rebalance dates
        self.rebalance_dates = self.p.rebalance_data["trade_date"].unique()
        if hasattr(self.rebalance_dates[0], "date"):
            self.rebalance_dates = [d.date() for d in self.rebalance_dates]
        
        self.rebalance_data = self.p.rebalance_data
        self.order_list = []  # Track pending orders
        self.previous_positions = []  # Track previous period holdings
    
    def log(self, txt, dt=None):
        """Strategy logging function."""
        if self.p.log_trades:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}: {txt}")
    
    def next(self):
        """Execute trades at current bar close (default mode)."""
        if self.p.sell_rule == "current-bar":
            self._execute_rebalancing_logic()
    
    def next_open(self):
        """Execute trades at next bar open (cheat-on-open mode)."""
        if self.p.sell_rule == "next-bar":
            self._execute_rebalancing_logic()
    
    def _execute_rebalancing_logic(self):
        """Core rebalancing logic shared between trading modes."""
        current_date = self.datas[0].datetime.date(0)
        
        if current_date not in self.rebalance_dates:
            return
        
        self.log(f"=== Rebalancing on {current_date} ===")
        self.log(f"Current portfolio value: {self.broker.getvalue():.2f}")
        
        # Cancel pending orders before rebalancing
        self._cancel_pending_orders()
        
        # Get target positions for current rebalance date
        current_targets = self._get_current_targets(current_date)
        target_instruments = current_targets["instrument"].tolist()
        
        # Close positions not in new target list
        self._close_unwanted_positions(target_instruments)
        
        # Open/adjust positions according to target weights
        self._execute_target_positions(current_targets)
        
        self.previous_positions = target_instruments
    
    def _cancel_pending_orders(self):
        """Cancel all pending orders before rebalancing."""
        if self.order_list:
            self.log("Canceling pending orders")
            for order in self.order_list:
                self.cancel(order)
            self.order_list.clear()
    
    def _get_current_targets(self, current_date) -> pd.DataFrame:
        """Get target positions for current rebalance date."""
        if isinstance(current_date, pd.Timestamp):
            mask = self.rebalance_data["trade_date"] == current_date
        else:
            mask = self.rebalance_data["trade_date"].dt.date == current_date
        return self.rebalance_data[mask]
    
    def _close_unwanted_positions(self, target_instruments: List[str]):
        """Close positions that are not in the new target list."""
        positions_to_close = [inst for inst in self.previous_positions 
                            if inst not in target_instruments]
        
        if positions_to_close:
            self.log(f"Closing positions: {positions_to_close}")
            for instrument in positions_to_close:
                try:
                    data = self.getdatabyname(instrument)
                    if self.getposition(data).size > 0:
                        order = self.close(data=data)
                        self.order_list.append(order)
                except:
                    self.log(f"Warning: Could not close position for {instrument}")
    
    def _execute_target_positions(self, targets: pd.DataFrame):
        """Execute target positions according to specified weights."""
        self.log("Executing target positions")
        
        for _, row in targets.iterrows():
            instrument = row["instrument"]
            weight = row["weight"]
            
            try:
                data = self.getdatabyname(instrument)
                # Reserve some cash to avoid insufficient funds
                target_weight = weight * (1 - self.p.cash_reserve)
                target_value = self.broker.getvalue() * target_weight
                
                # Use appropriate price based on trading mode
                if self.p.sell_rule == "next-bar":
                    # Cheat-on-open mode: use open price
                    price = data.open[0]
                else:
                    # Default mode: use close price
                    price = data.close[0]
                
                target_shares = target_value / price
                current_shares = self.getposition(data).size
                trade_size = target_shares - current_shares
                trade_size = trade_size // self.p.trade_unit * self.p.trade_unit
                
                if abs(trade_size) >= self.p.trade_unit:  # Only trade if meets minimum unit
                    if trade_size > 0:
                        order = self.buy(data, size=abs(trade_size))
                    else:
                        order = self.sell(data, size=abs(trade_size))
                    self.order_list.append(order)
                    
            except Exception as e:
                self.log(f"Warning: Could not place order for {instrument}: {str(e)}")
    
    def notify_order(self, order):
        """Handle order execution notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{action} EXECUTED - {order.data._name}: "
                f"Price: {order.executed.price:.2f}, "
                f"Size: {order.executed.size:.0f}, "
                f"Cost: {order.executed.value:.2f}, "
                f"Comm: {order.executed.comm:.2f}"
            )


class FactorSelectionStrategy(bt.Strategy):
    """Multi-factor stock selection strategy with real-time factor calculation.
    
    This strategy re-implements factor calculation logic within Backtrader
    and performs stock selection based on factor rankings on rebalance dates.
    """
    
    params = dict(
        selection_num=10,         # Number of stocks to hold
        factor_ascending=False,   # Factor sorting direction: False=higher is better
        rebalance_dates=None,     # List of rebalance dates
        cash_reserve=0.05,        # Reserve cash percentage
        trade_unit=100,           # Minimum order number for each trade
        log_trades=True,          # Whether to log trade executions
        sell_rule="current-bar",  # Trading timing rule
    )
    
    def __init__(self):
        """Initialize the factor selection strategy."""
        if self.p.rebalance_dates is None:
            raise ValueError("Rebalance dates must be provided")
        
        # Calculate target weight per position
        self.target_weight = 1.0 / self.p.selection_num
        
        # Extract factor data from each data feed
        self.factor_data = {data: data.lines.factor for data in self.datas if data._name != "benchmark"}
        
        # Convert rebalance dates to set for faster lookup
        self.rebalance_set = set()
        for date in self.p.rebalance_dates:
            if isinstance(date, pd.Timestamp):
                self.rebalance_set.add(date.date())
            else:
                self.rebalance_set.add(date)
        
        self.last_rebalance = None
    
    def log(self, txt, dt=None):
        """Strategy logging function."""
        if self.p.log_trades:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}: {txt}")
    
    def is_rebalance_day(self, current_date) -> bool:
        """Check if current date is a rebalance day."""
        if self.last_rebalance is None:
            is_rebalance = current_date in self.rebalance_set
        else:
            is_rebalance = (current_date in self.rebalance_set and 
                          current_date != self.last_rebalance)
        
        if is_rebalance:
            self.last_rebalance = current_date
        
        return is_rebalance
    
    def next(self):
        """Execute trades at current bar close (default mode)."""
        if self.p.sell_rule == "current-bar":
            self._execute_factor_selection_logic()
    
    def next_open(self):
        """Execute trades at next bar open (cheat-on-open mode)."""
        if self.p.sell_rule == "next-bar":
            self._execute_factor_selection_logic()
    
    def _execute_factor_selection_logic(self):
        """Core factor selection logic shared between trading modes."""
        current_date = self.datas[0].datetime.date(0)
        
        if not self.is_rebalance_day(current_date):
            return
        
        self.log(f"=== Factor-based rebalancing on {current_date} ===")
        self.log(f"Current portfolio value: {self.broker.getvalue():.2f}")
        
        # Get valid stocks with factor values
        valid_stocks = self._get_valid_stocks()
        
        if not valid_stocks:
            self.log("Warning: No valid stocks with factor values found")
            return
        
        # Rank stocks by factor values
        selected_stocks = self._select_top_stocks(valid_stocks)
        
        # Get current positions
        current_positions = [data for data, pos in self.getpositions().items() if pos]
        
        # Close positions not in new selection
        self._close_unwanted_positions(current_positions, selected_stocks)
        
        # Open/adjust positions for selected stocks
        self._execute_selected_positions(selected_stocks)
    
    def _get_valid_stocks(self) -> List[bt.DataBase]:
        """Get stocks with valid (non-NaN) factor values."""
        valid_stocks = []
        for data in self.datas:
            try:
                factor_value = self.factor_data[data][0]  # Current factor value
                if not bt.math.isnan(factor_value):
                    valid_stocks.append(data)
            except (IndexError, KeyError):
                continue
        return valid_stocks
    
    def _select_top_stocks(self, valid_stocks: List[bt.DataBase]) -> List[bt.DataBase]:
        """Select top stocks based on factor ranking."""
        # Sort stocks by factor values
        sorted_stocks = sorted(
            valid_stocks,
            key=lambda x: self.factor_data[x][0],
            reverse=not self.p.factor_ascending
        )
        
        # Select top N stocks
        selected_count = min(self.p.selection_num, len(sorted_stocks))
        selected_stocks = sorted_stocks[:selected_count]
        
        selected_names = [stock._name for stock in selected_stocks]
        self.log(f"Selected stocks: {selected_names}")
        
        return selected_stocks
    
    def _close_unwanted_positions(self, current_positions: List[bt.DataBase], 
                                 selected_stocks: List[bt.DataBase]):
        """Close positions not in the new selection."""
        positions_to_close = set(current_positions) - set(selected_stocks)
        
        for data in positions_to_close:
            self.log(f"Closing position: {data._name}")
            self.order_target_percent(data, target=0.0)
    
    def _execute_selected_positions(self, selected_stocks: List[bt.DataBase]):
        """Execute positions for selected stocks."""
        target_weight = self.target_weight * (1 - self.p.cash_reserve)
        
        for data in selected_stocks:
            self.log(f"Adjusting position: {data._name} to {target_weight:.1%}")
            target_value = self.broker.getvalue() * target_weight
            
            # Use appropriate price based on trading mode
            if self.p.sell_rule == "next-bar":
                # Cheat-on-open mode: use open price
                price = data.open[0]
            else:
                # Default mode: use close price
                price = data.close[0]
            
            target_shares = target_value / price
            current_shares = self.getposition(data).size
            trade_size = target_shares - current_shares
            trade_size = trade_size // self.p.trade_unit * self.p.trade_unit
            
            if abs(trade_size) >= self.p.trade_unit:  # Only trade if meets minimum unit
                if trade_size > 0:
                    self.buy(data, size=abs(trade_size))
                else:
                    self.sell(data, size=abs(trade_size))
    
    def notify_order(self, order):
        """Handle order execution notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            action = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{action} EXECUTED - {order.data._name}: "
                f"Price: {order.executed.price:.2f}, "
                f"Size: {order.executed.size:.0f}, "
                f"Cost: {order.executed.value:.2f}, "
                f"Comm: {order.executed.comm:.2f}"
            )


class Frozen2Backtrader:
    """Converter class to transform Frozen strategies into Backtrader-compatible format.
    
    This class provides two main approaches for strategy conversion:
    1. Rebalancing table-based: Uses pre-calculated portfolio weights
    2. Factor selection-based: Re-implements factor calculation logic in Backtrader

    The wrapper supports two trading modes:
    - "next-bar": Cheat-on-open mode (trades at next bar's open)
    - "current-bar": Default mode (trades at current bar's close, executed at next open)
    """

    def __init__(self, frozen_strategy: FrozenBt, mode: str = "rebalance_table"):
        """Initialize the Frozen-to-Backtrader converter.
        
        Args:
            frozen_strategy: The Frozen strategy instance to convert
            mode: Conversion mode - 'rebalance_table' or 'factor_selection'
        """
        self.frozen_strategy = frozen_strategy
        self.config = frozen_strategy.config
        self.mode = mode
        
        # Initialize components
        self._prepare_datafeed()
        self._create_cerebro()
        self._set_params()
        
        # Mode-specific initialization
        if mode == "rebalance_table":
            self._prepare_rebalance_table()
        elif mode == "factor_selection":
            self._prepare_factor_selection()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'rebalance_table' or 'factor_selection'")
        
        self._feed_market_data()
    
    def _prepare_datafeed(self):
        """Prepare all market data for Backtrader consumption."""
        start_date = self.frozen_strategy.start_date
        end_date = self.frozen_strategy.end_date
        
        self._ticker_datafeed(start_date, end_date)
        self._benchmark_datfeed(start_date, end_date)
    
    def _ticker_datafeed(self, start_date, end_date):
        """Prepare ticker-specific data for Backtrader consumption."""
        pool = self.frozen_strategy.universe
        
        # Load bar data (ticker specific)
        self.ticker_data = self.frozen_strategy.dataloader.load_volume_price(
            "stock_daily_real", 
            ("open", "high", "low", "close", "volume"), 
            pool, start_date, end_date, multiindex=True
        )
        self.ticker_data["openinterest"] = 0
        self.ticker_data.index.names = ["ticker", "datetime"]
    
    def _benchmark_datfeed(self, start_date, end_date):
        """Prepare benchmark data for Backtrader consumption."""
        benchmark_data = self.frozen_strategy.dataloader.load_volume_price(
            "index_daily",
            ("open", "high", "low", "close", "volume"),
            (self.config.benchmark,), start_date, end_date, multiindex=True)
        benchmark_data["openinterest"] = 0
        benchmark_data.index.names = ["ticker", "datetime"]
        self.benchmark_data = benchmark_data.xs(self.config.benchmark)
    
    def _create_cerebro(self):
        """Initialize Backtrader Cerebro engine."""
        # Set cheat_on_open based on sell_rule configuration
        cheat_on_open = (getattr(self.config, "sell_rule") == "next-bar")
        self.cerebro = bt.Cerebro(cheat_on_open=cheat_on_open)
    
    def _set_params(self):
        """Configure Backtrader broker parameters from Frozen config."""
        # Set initial capital
        self.cerebro.broker.setcash(self.config.init_capital)
        
        # Set commission
        self._customize_commission()
        
        # Set slippage
        if hasattr(self.config, "slippage_type"):
            if self.config.slippage_type == "percentage":
                self.cerebro.broker.set_slippage_perc(perc=self.config.slippage_rate)
            else:  # absolute
                self.cerebro.broker.set_slippage_fixed(fixed=self.config.slippage_rate)
        
        # Set order sizer
        self.cerebro.addsizer(bt.sizers.FixedSize, stake=self.config.trade_unit)

        # Set order filler
        self.cerebro.broker.set_filler(bt.broker.fillers.FixedBarPerc(perc=50))
    
    def _customize_commission(self):
        """Customized transaction fee for Chinese stock market."""
        class StockCommission(bt.CommInfoBase):
            params = (
                ("stocklike", True),
                ("commtype", bt.CommInfoBase.COMM_PERC),
                ("percabs", True),
                ("stamp_duty", self.config.stamp_duty),
            )
            
            def _getcommission(self, size, price, pseudoexec):
                if size > 0:
                    return abs(size) * price * self.p.commission
                elif size < 0:
                    return abs(size) * price * (self.p.commission + self.p.stamp_duty)
                else:
                    return 0
        comminfo = StockCommission(commission=self.config.commission)
        self.cerebro.broker.addcommissioninfo(comminfo)
    
    def _prepare_rebalance_table(self):
        """Prepare rebalancing table data for table-based strategy."""
        # Look for rebalancing table in strategy output directory
        report_dir = Path(self.config.strategy_folder) / "performance_report"
        rebalance_file = report_dir / "instrument orders.xlsx"
        
        if rebalance_file.exists():
            rebalance_data = pd.read_excel(rebalance_file, index_col=[0, 1])
        else:
            # Try to generate rebalance table from frozen strategy
            try:
                # Run frozen strategy to get results
                result_package = self.frozen_strategy.run_backtest(plot=False, print_table=False, send_backend=True)
                
                # Convert to rebalance table format
                rebalance_data = result_package["order_history"]
            except Exception as e:
                raise ValueError(f"Could not prepare rebalance table: {str(e)}. "
                               f"Please ensure the strategy has been run at least once to generate order history.")
        
        if self.config.sell_rule == "current-bar":
            # Transform table dates to actual rebalance dates
            unique_dates = rebalance_data.index.get_level_values(0).unique()
            rebalance_dates = self.frozen_strategy.calendar.previous_trade_dates(unique_dates)
            mapping = dict(zip(unique_dates, rebalance_dates))
            new_level0 = rebalance_data.index.get_level_values(0).map(mapping)
            new_index = pd.MultiIndex.from_arrays(
                [new_level0, rebalance_data.index.get_level_values(1)],
                names=rebalance_data.index.names
            )
            rebalance_data.index = new_index
        
        # Ensure correct dataframe structure
        self.rebalance_data = rebalance_data.reset_index()
        self.rebalance_data.columns = ["trade_date", "instrument", "weight"]
    
    def _prepare_factor_selection(self):
        """Prepare factor data for factor selection strategy."""
        # Calculate factor values
        self.factor_data = self.frozen_strategy.calc()
        
        # Get rebalance frequency from config (use date_rule as default)
        rebalance_freq = getattr(self.config, "date_rule")
        
        # Prepare factor data aligned with market data dates
        factor_start = self.frozen_strategy.start_date
        factor_end = self.frozen_strategy.end_date
        
        self.aligned_factor, self.rebalance_dates = self._prepare_factor_data(
            self.factor_data.data[factor_start:factor_end],
            freq=rebalance_freq
        )
    
    def _prepare_factor_data(self, factor_df: pd.DataFrame, freq: str):
        """Convert daily factor data to specified rebalance frequency.
        
        Args:
            factor_df: Daily factor DataFrame (index=date, columns=tickers)
            freq: Resampling frequency (e.g., 'W-WED', '2W-WED', 'ME', '2ME', 'QS')
            
        Returns:
            tuple: (aligned_factor, rebalance_dates)
        """
        # Resample to specified frequency
        resampled_factor = factor_df.resample(freq).mean()
        
        # Get rebalance dates (last trading day of each period)
        rebalance_dates = []
        for _, frame in factor_df.groupby(pd.Grouper(freq=freq)):
            if not frame.empty:
                last_trade_date = frame.index[-1]
                rebalance_dates.append(last_trade_date)
        
        # Forward fill factor values to daily frequency
        aligned_factor = resampled_factor.reindex(factor_df.index, method="ffill")
        
        if self.config.sell_rule == "next-bar":
            # Get next trading dates for rebalancing
            rebalance_dates = self.frozen_strategy.calendar.next_trade_dates(rebalance_dates)
        
        return aligned_factor, rebalance_dates
    
    def _feed_market_data(self):
        """Feed market data into Cerebro engine."""
        tickers = self.ticker_data.index.get_level_values("ticker").unique()
        
        # Feed ticker data
        for ticker in tqdm(tickers, desc="Loading market data"):
            ticker_data = self.ticker_data.xs(ticker)
            
            if self.mode == "factor_selection":
                # Add factor data as additional line
                ticker_data["factor"] = self.aligned_factor[ticker]
                datafeed = PandasDataWithFactor(
                    dataname=ticker_data,
                    fromdate=pd.Timestamp(self.frozen_strategy.start_date),
                    todate=pd.Timestamp(self.frozen_strategy.end_date)
                )
            else:
                # Standard data feed for rebalance table mode
                datafeed = bt.feeds.PandasData(
                    dataname=ticker_data,
                    fromdate=pd.Timestamp(self.frozen_strategy.start_date),
                    todate=pd.Timestamp(self.frozen_strategy.end_date)
                )
            
            self.cerebro.adddata(datafeed, name=ticker)
        
        # Feed benchmark data
        benchmark_datafeed = bt.feeds.PandasData(
            dataname=self.benchmark_data,
            fromdate=pd.Timestamp(self.frozen_strategy.start_date),
            todate=pd.Timestamp(self.frozen_strategy.end_date)
        )
        self.cerebro.adddata(benchmark_datafeed, name="benchmark")
        self.cerebro.benchmark_data = self.benchmark_data

        # Add analyzer for benchmark returns
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, data=benchmark_datafeed, _name="_BenchmarkReturn")
        
        # Add observer for time returns
        self.cerebro.addobserver(bt.observers.Benchmark, data=benchmark_datafeed)
        self.cerebro.addobserver(bt.observers.TimeReturn)

    
    def add_strategy(self, **strategy_params):
        """Add appropriate strategy to Cerebro based on conversion mode.
        
        Args:
            **strategy_params: Additional parameters to pass to the strategy
        """
        # Get parameters from config
        trade_unit = getattr(self.config, "trade_unit")
        sell_rule = getattr(self.config, "sell_rule")
        
        if self.mode == "rebalance_table":
            default_params = {
                "rebalance_data": self.rebalance_data,
                "trade_unit": trade_unit,
                "sell_rule": sell_rule,
            }
            default_params.update(strategy_params)
            
            self.cerebro.addstrategy(
                RebalanceTableStrategy,
                **default_params
            )
        elif self.mode == "factor_selection":
            # Extract strategy parameters from config
            asset_range = getattr(self.config, "asset_range")

            # For asset range (a, b), the latter should be greater than 1
            selection_num = asset_range[1]
            
            # Determine factor sorting direction based on strategy
            # Default to False (higher values are better) for momentum strategies
            factor_ascending = False
            
            default_params = {
                "rebalance_dates": self.rebalance_dates,
                "selection_num": selection_num,
                "trade_unit": trade_unit,
                "factor_ascending": factor_ascending,
                "sell_rule": sell_rule,
            }
            default_params.update(strategy_params)
            
            self.cerebro.addstrategy(
                FactorSelectionStrategy,
                **default_params
            )
    
    def add_analyzers(self):
        """Add standard analyzers to Cerebro."""
        analyzers = [
            (bt.analyzers.TimeReturn, "_TimeReturn"),
            (bt.analyzers.AnnualReturn, "_AnnualReturn"),
            (bt.analyzers.SharpeRatio, "_SharpeRatio"),
            (bt.analyzers.DrawDown, "_DrawDown")
        ]

        from .extensions.addon import TradeListAnalyzer, KeyIndicatorAnalyzer
        addon_analyzers = [
            (TradeListAnalyzer, "_TradeList"),
            (KeyIndicatorAnalyzer, "_KeyIndicator")
        ]
        analyzers += addon_analyzers
        
        # Add PyFolio analyzer if available
        try:
            analyzers.append((bt.analyzers.PyFolio, "_PyFolio"))
        except AttributeError:
            pass  # PyFolio analyzer not available in this Backtrader version
        
        for analyzer_cls, name in analyzers:
            if analyzer_cls == bt.analyzers.SharpeRatio:
                # Use risk-free rate from config if available
                risk_free_rate = 0.00
                self.cerebro.addanalyzer(
                    analyzer_cls, 
                    riskfreerate=risk_free_rate,
                    _name=name,
                    timeframe=bt.TimeFrame.Days,
                    compression=1,
                    factor=252,
                    annualize=True
                )
            else:
                self.cerebro.addanalyzer(analyzer_cls, _name=name)
    
    def add_observers(self):
        """Add standard observers to Cerebro."""
        self.cerebro.addobserver(bt.observers.Value)  # Portfolio value tracking
        self.cerebro.addobserver(bt.observers.Trades)  # Trade tracking
    
    def run(self, add_analyzers: bool = True, add_observers: bool = True, **strategy_params):
        """Run the backtest with converted strategy.
        
        Args:
            add_analyzers: Whether to add standard analyzers
            add_observers: Whether to add standard observers
            **strategy_params: Additional parameters for the strategy
            
        Returns:
            Backtrader results object
        """
        # Add strategy
        self.add_strategy(**strategy_params)
        
        # Add analyzers and observers if requested
        if add_analyzers:
            self.add_analyzers()
        if add_observers:
            self.add_observers()
        
        # Run backtest
        results = self.cerebro.run(tradehistory=True)
        
        return results
    
    def plot(self, **plot_kwargs):
        """Plot backtest results.
        
        Args:
            **plot_kwargs: Arguments to pass to cerebro.plot()
        """
        self.cerebro.plot(**plot_kwargs)


class Frozen2VectorBT:
    """Converter class to transform Frozen strategies into VectorBT-compatible format.
    
    This class provides two main approaches for strategy conversion:
    1. Rebalancing table-based: Uses pre-calculated portfolio weights  
    2. Factor selection-based: Re-implements factor calculation logic in VectorBT
    
    The wrapper supports vectorbt's multi-asset portfolio backtesting capabilities.
    """

    def __init__(self, frozen_strategy: FrozenBt, mode: str = "rebalance_table"):
        """Initialize the Frozen-to-VectorBT converter.
        
        Args:
            frozen_strategy: The Frozen strategy instance to convert
            mode: Conversion mode - 'rebalance_table' or 'factor_selection'
        """
        self.frozen_strategy = frozen_strategy
        self.config = frozen_strategy.config
        self.mode = mode
        
        # Initialize components
        self._prepare_market_data()
        
        # Mode-specific initialization
        if mode == "rebalance_table":
            self._prepare_rebalance_table()
        elif mode == "factor_selection":
            self._prepare_factor_selection()
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'rebalance_table' or 'factor_selection'")
    
    def _prepare_market_data(self):
        """Prepare market data for VectorBT consumption."""
        start_date = self.frozen_strategy.start_date
        end_date = self.frozen_strategy.end_date
        pool = self.frozen_strategy.universe
        
        # Load bar data
        self.ticker_data = self.frozen_strategy.dataloader.load_volume_price(
            "stock_daily_real", 
            ("open", "high", "low", "close", "volume"), 
            pool, start_date, end_date, multiindex=True
        )
        
        # Prepare close prices for vectorbt (columns=tickers, index=dates)
        self.close_prices = self.ticker_data["close"].unstack(level=0)
        self.open_prices = self.ticker_data["open"].unstack(level=0)
        self.close_prices.columns.name = None
        self.open_prices.columns.name = None
        
        self.trade_price = self.open_prices if self.config.sell_rule == "next-bar" else self.close_prices
        
        # Load benchmark data if available
        try:
            benchmark_data = self.frozen_strategy.dataloader.load_volume_price(
                "index_daily",
                ("close",),
                (self.config.benchmark,), start_date, end_date, multiindex=True
            )
            self.benchmark_returns = benchmark_data["close"].xs(self.config.benchmark).pct_change().fillna(0)
        except:
            self.benchmark_returns = None
    
    def _prepare_rebalance_table(self):
        """Prepare rebalancing table data for table-based strategy."""
        # Look for rebalancing table in strategy output directory
        report_dir = Path(self.config.strategy_folder) / "performance_report"
        rebalance_file = report_dir / "instrument orders.xlsx"
        
        if rebalance_file.exists():
            rebalance_data = pd.read_excel(rebalance_file, index_col=[0, 1])
        else:
            # Try to generate rebalance table from frozen strategy
            try:
                # Run frozen strategy to get results
                result_package = self.frozen_strategy.run_backtest(plot=False, print_table=False, send_backend=True)
                
                # Convert to rebalance table format
                rebalance_data = result_package["order_history"]
            except Exception as e:
                raise ValueError(f"Could not prepare rebalance table: {str(e)}. "
                               f"Please ensure the strategy has been run at least once to generate order history.")
        
        if self.config.sell_rule == "current-bar":
            # Transform table dates to actual rebalance dates
            unique_dates = rebalance_data.index.get_level_values(0).unique()
            rebalance_dates = self.frozen_strategy.calendar.previous_trade_dates(unique_dates)
            mapping = dict(zip(unique_dates, rebalance_dates))
            new_level0 = rebalance_data.index.get_level_values(0).map(mapping)
            new_index = pd.MultiIndex.from_arrays(
                [new_level0, rebalance_data.index.get_level_values(1)],
                names=rebalance_data.index.names
            )
            rebalance_data.index = new_index
        
        # Convert to vectorbt-compatible format
        self.rebalance_data = rebalance_data.reset_index()
        self.rebalance_data.columns = ["trade_date", "instrument", "weight"]
        
        # Create target allocation matrix (dates x instruments)
        self.target_weights = self.rebalance_data.pivot(
            index="trade_date", columns="instrument", values="weight"
        )
        
        # Reindex to match price data dates and columns
        # For time dimension: use ffill to forward fill weights
        # For column dimension: use 0 for new instruments not in rebalance data
        self.target_weights = self.target_weights.reindex(
            index=self.close_prices.index, 
            method="ffill"
        ).reindex(
            columns=self.close_prices.columns,
            fill_value=0
        ).fillna(0)
    
    def _prepare_factor_selection(self):
        """Prepare factor data for factor selection strategy."""
        # Calculate factor values
        factor = self.frozen_strategy.calc()
        
        # Prepare factor data aligned with market data dates
        factor_start = self.frozen_strategy.start_date
        factor_end = self.frozen_strategy.end_date
        
        self.factor_data = factor.data[factor_start:factor_end]
        
        # Get rebalance frequency from config
        rebalance_freq = getattr(self.config, "date_rule")

        self.resampled_factor, self.rebalance_dates = self._prepare_factor_data(
            self.factor_data,
            freq=rebalance_freq
        )
        
        # Extract strategy parameters from config
        asset_range = getattr(self.config, "asset_range")
        self.selection_num = asset_range[1]
    
    def _prepare_factor_data(self, factor_df: pd.DataFrame, freq: str):
        """Convert daily factor data to specified rebalance frequency.
        
        Args:
            factor_df: Daily factor DataFrame (index=date, columns=tickers)
            freq: Resampling frequency (e.g., 'W-WED', '2W-WED', 'ME', '2ME', 'QS')
            
        Returns:
            tuple: (aligned_factor, rebalance_dates)
        """
        # Resample to specified frequency
        resampled_factor = factor_df.resample(freq, label="right").mean()
        
        # Get rebalance dates (last trading day of each period)
        rebalance_dates = []
        for _, frame in factor_df.groupby(pd.Grouper(freq=freq)):
            if not frame.empty:
                last_trade_date = frame.index[-1]
                rebalance_dates.append(last_trade_date)
        
        if self.config.sell_rule == "next-bar":
            # Get next trading dates for rebalancing
            rebalance_dates = self.frozen_strategy.calendar.next_trade_dates(rebalance_dates)
        
        return resampled_factor, rebalance_dates
    
    def _select_top_stocks(self, factor_values: pd.Series, n: int) -> pd.Series:
        """Select top n stocks based on factor ranking."""
        # Rank stocks by factor values (higher is better by default)
        ranked = factor_values.rank(axis=1, ascending=False, method="first")
        # Return boolean mask for top n stocks
        return ranked <= n
    
    def run_rebalance_table_backtest(self, **kwargs) -> vbt.Portfolio:
        """Run backtest using rebalancing table approach.
        
        Args:
            **kwargs: Additional parameters for Portfolio.from_orders
            
        Returns:
            VectorBT Portfolio object
        """
        if self.mode != "rebalance_table":
            raise ValueError("This method requires 'rebalance_table' mode")
        
        # Use target weights as size (target percentage allocation)
        pf = vbt.Portfolio.from_orders(
            close=self.trade_price,
            size=self.target_weights,
            size_type="targetpercent",
            fees=self.config.commission,
            slippage=self.config.slippage_rate,
            init_cash=self.config.init_capital,
            min_size=self.config.trade_unit,
            size_granularity=self.config.trade_unit,
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            **kwargs
        )

        # Use fixed cash determined by weights as size
        # pf = vbt.Portfolio.from_signals(
        #     close=self.open_prices,
        #     entries=self.target_weights.astype(bool),
        #     exits=~self.target_weights.astype(bool),
        #     size=self.target_weights * self.config.init_capital * 0.95,  # The target percent allocation for each asset
        #     size_type="value",
        #     fees=self.config.commission,
        #     slippage=self.config.slippage_rate,
        #     init_cash=self.config.init_capital,
        #     min_size=self.config.trade_unit,
        #     size_granularity=self.config.trade_unit,
        #     group_by=True,
        #     cash_sharing=True,
        #     call_seq="auto",
        #     **kwargs
        # )
        
        return pf
    
    def run_factor_selection_backtest(self, **kwargs) -> vbt.Portfolio:
        """Run backtest using factor selection approach.
        
        Args:
            **kwargs: Additional parameters for Portfolio.from_signals
            
        Returns:
            VectorBT Portfolio object
        """
        if self.mode != "factor_selection":
            raise ValueError("This method requires 'factor_selection' mode")
        
        # Create top stocks selection based on factor rankings
        selected = self._select_top_stocks(self.resampled_factor, self.selection_num)
        signals = pd.DataFrame(index=self.rebalance_dates, columns=selected.columns)
        for rb_date in self.rebalance_dates:
            if self.config.sell_rule == "current-bar":
                valid_dates = selected.index[selected.index >= rb_date]
                nearest_date = valid_dates[0]
            else:
                valid_dates = selected.index[selected.index <= rb_date]
                nearest_date = valid_dates[-1]
            signals.loc[rb_date] = selected.loc[nearest_date]
        
        signals = signals.reindex(self.factor_data.index, method="ffill", fill_value=False)
        
        # Create target weights: equal weight for selected stocks, 0 for others
        selected_counts = signals.sum(axis=1)
        target_weights = pd.DataFrame(0, index=signals.index, columns=signals.columns)
        non_zero_rows = selected_counts != 0
        target_weights.loc[non_zero_rows] = signals.loc[non_zero_rows].div(selected_counts[non_zero_rows], axis=0)
        
        self.target_weights = target_weights.fillna(0)
        self.target_weights.columns.name = None

        # Fix index dtype mismatch between price and weights data
        # Reindex target_weights to match trade_price index exactly
        self.target_weights = self.target_weights.reindex(
            self.trade_price.index, method="ffill", fill_value=0
        )

        # Run portfolio backtest with target weights
        pf = vbt.Portfolio.from_orders(
            close=self.trade_price,
            size=self.target_weights,
            size_type="targetpercent", 
            init_cash=self.config.init_capital,
            fees=self.config.commission,
            slippage=self.config.slippage_rate,
            min_size=self.config.trade_unit,
            size_granularity=self.config.trade_unit,
            group_by=True,
            cash_sharing=True,
            call_seq="auto",
            **kwargs
        )
        
        return pf
    
    def run_backtest(self, **kwargs) -> vbt.Portfolio:
        """Run backtest based on the selected mode.
        
        Args:
            **kwargs: Parameters to pass to the specific backtest method
            
        Returns:
            VectorBT Portfolio object
        """
        if self.mode == "rebalance_table":
            return self.run_rebalance_table_backtest(**kwargs)
        elif self.mode == "factor_selection":
            return self.run_factor_selection_backtest(**kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def compare_with_benchmark(self, portfolio: vbt.Portfolio) -> pd.DataFrame:
        """Compare portfolio performance with benchmark.
        
        Args:
            portfolio: VectorBT Portfolio object
            
        Returns:
            DataFrame with performance comparison
        """
        if self.benchmark_returns is None:
            return None
        
        # Get portfolio returns
        portfolio_returns = portfolio.returns()
        if hasattr(portfolio_returns, "grouper") and portfolio_returns.grouper is not None:
            # For grouped portfolios, get total returns
            portfolio_returns = portfolio_returns.sum(axis=1)
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(self.benchmark_returns.index)
        portfolio_rets = portfolio_returns.loc[common_dates]
        benchmark_rets = self.benchmark_returns.loc[common_dates]
        
        # Calculate cumulative returns
        portfolio_cumret = (1 + portfolio_rets).cumprod()
        benchmark_cumret = (1 + benchmark_rets).cumprod()
        
        comparison = pd.DataFrame({
            "Portfolio": portfolio_cumret,
            "Benchmark": benchmark_cumret,
            "Excess": portfolio_cumret - benchmark_cumret
        })
        
        return comparison


class PerformanceAnalyzer(ABC):
    """Abstract base class for performance analysis across different backtesting engines.
    
    This class defines the interface for performance analysis and visualization
    capabilities. Each engine should have its own concrete implementation.
    """
    
    def __init__(self, result=None, config=None):
        """Initialize the performance analyzer.
        
        Args:
            external_strategy: Strategy that's wrapped by external engines
            result: Result object from the backtest engine
            config: Configuration object containing engine settings
        """
        self.result = result[0] if isinstance(result, (list, tuple)) else result
        self.config = config

        L = FrozenLogger(config, run_mode="normal")
        self.FULL_PATH = L.FULL_PATH
    
    def analyze(self, 
               plot: bool = True, 
               plot_type: str = "line",
               dynamic: bool = False,
               print_table: bool = True,
               positions: bool = False,
               trades: bool = False,
               html: bool = False,
               send_backend: bool = False):
        """Perform comprehensive performance analysis.
        
        Args:
            plot: Whether to generate plots
            plot_type: Type of plot ('line', 'scatter', etc.)
            dynamic: Whether to generate dynamic plots
            print_table: Whether to print performance metrics table
            positions: Whether to show positions
            trades: Whether to show trades
            html: Whether to generate HTML output
            send_backend: Whether to send results to backend
        
        Returns:
            Portfolio returns or dictionary containing analysis results
        """
        from ...utils.report import Report
        analysis_results = {}
        
        # Get performance metrics
        if print_table:
            metrics = self.get_metrics()
            self._print_performance_table(metrics)
            analysis_results["metrics"] = metrics
        
        # Generate plots
        if plot:
            self.generate_plot(plot_type, dynamic)
        
        # Get portfolio returns
        analysis_results["returns"] = self.returns
        
        # Get positions
        if positions:
            positions = self.get_positions()
            Report.to_excel(positions, self.FULL_PATH, type="positions")
            analysis_results["positions"] = positions
        
        # Get trades
        if trades:
            trades = self.get_trades()
            Report.to_excel(trades, self.FULL_PATH, type="trades")
            analysis_results["trades"] = trades
        
        # Generate HTML output
        if html:
            Report.to_quantstats(self.returns, self.FULL_PATH)
        
        if send_backend:
            return analysis_results
        else:
            return self.returns
    
    @abstractmethod
    def get_metrics(self):
        """Extract performance metrics from the engine result.
        
        Returns:
            Dictionary containing performance metrics
        """
        pass
    
    @abstractmethod
    def generate_plot(self, plot_type="line", dynamic=False):
        """Generate performance plots.
        
        Args:
            plot_type: Type of plot ('line', 'scatter', etc.)
            dynamic: Whether to generate dynamic plots
            
        Returns:
            Plot object or reference
        """
        pass
    
    @abstractmethod
    def get_positions(self):
        """Get position information from the engine result.
        
        Returns:
            Position data structure
        """
        pass
    
    @abstractmethod
    def get_trades(self):
        """Get trade information from the engine result.
        
        Returns:
            Trade data structure
        """
        pass
    
    def generate_html(self):
        """Generate HTML output. Default implementation returns None.
        
        Returns:
            HTML output or None
        """
        return None
    
    def _print_performance_table(self, metrics):
        """Print performance metrics in a formatted table."""
        if metrics is None:
            print("No metrics available")
            return
        
        metrics_df = pd.Series(metrics)
        print(metrics_df)
        return metrics_df

class BacktraderAnalyzer(PerformanceAnalyzer):
    """Performance analyzer for Backtrader engine results."""
    
    def get_metrics(self):
        """Extract metrics from backtrader result."""
        metrics = {}
        
        if self.result is None:
            return metrics
        
        try:
            if hasattr(self.result, "analyzers"):
                # Get relevant analyzers
                # annual_ret = self.result.analyzers._AnnualReturn.get_analysis()
                # sharpe_ratio = self.result.analyzers._SharpeRatio.get_analysis()
                # max_drawdown = self.result.analyzers._DrawDown.get_analysis()

                # metrics["annual_ret"] = annual_ret
                # metrics["sharpe_ratio"] = sharpe_ratio
                # metrics["max_drawdown"] = max_drawdown

                key_indicator = self.result.analyzers._KeyIndicator.get_analysis_data(self.result.cerebro.benchmark_data, "benchmark")[0]
                metrics = key_indicator.iloc[0, 1:].to_dict()
                metrics["Benchmark_Return"] = key_indicator.iloc[1]["Total_Return"]
            
            # Get basic portfolio value
            if hasattr(self.result, "broker"):
                metrics["Final_Value"] = self.result.broker.getvalue()
                metrics["Cash"] = self.result.broker.getcash()
        
        except Exception as e:
            print(f"Error extracting backtrader metrics: {e}")
        
        # Standardize table names
        ordered_metrics = OrderedDict([
            ("Strategy Return", metrics["Total_Return"]),
            ("Benchmark Return", metrics["Benchmark_Return"]),
            ("Annualized Return", metrics["Annualized_Return"]),
            ("Final Value", metrics["Final_Value"]),
            ("Cash", metrics["Cash"]),
            ("Max Drawdown", metrics["Max_Drawdown"]),
            ("Win Rate", metrics["Win_Rate"]),
            ("Sharpe Ratio", metrics["Sharpe_Ratio"]),
            ("Kelly Ratio", metrics["Kelly_Ratio"]),
            ("Recent 7D Return", metrics["Recent_7D_Return"]),
            ("Recent 30D Return", metrics["Recent_30D_Return"]),
            ("Commission Ratio", metrics["Commission_Ratio"]),
            ("Total Trades", metrics["Total_Trades"])
        ])
        return ordered_metrics
    
    def generate_plot(self, plot_type="line", dynamic=False):
        """Generate plots for backtrader results."""
        if self.result is None:
            return None
            
        try:
            if hasattr(self.result, "analyzers"):
                port_ret = pd.Series(self.result.analyzers._TimeReturn.get_analysis())
                benchmark_ret = pd.Series(self.result.analyzers._BenchmarkReturn.get_analysis())
            
            self.returns = pd.concat([port_ret, benchmark_ret], axis=1)
            self.returns.columns = ["Account", "Benchmark"]
            perf_plot.create_pnl_plot(self.returns, self.config.date_rule, plot_type, dynamic)
        
        except Exception as e:
            print(f"Error plotting backtrader results: {e}")
    
    def get_positions(self):
        """Get positions from backtrader results."""
        if self.result is None:
            return {}
            
        self.positions = []
        try:
            positions_dict = self.result.getpositions()
            for data_feed, position in positions_dict.items():
                if position.size != 0:
                    ticker = data_feed._name
                    position_value = position.size * position.price if position.size != 0 else 0
                    position_direction = "Long" if position.size > 0 else "Short"
                    self.positions.append({
                        "ticker": ticker,
                        "size": position.size,
                        "price": position.price,
                        "price_orig": position.price_orig,
                        "position_value": position_value,
                        "direction": position_direction,
                        "upopened": position.upopened,
                        "upclosed": position.upclosed,
                        "adjbase": position.adjbase,
                        "datetime": getattr(position, "datetime", None)
                    })
        except Exception as e:
            print(f"Error extracting backtrader positions: {e}")
            
        return pd.DataFrame(self.positions)
    
    def get_trades(self):
        """Get trades from backtrader results."""
        if self.result is None:
            return {}
            
        self.trades = {}
        try:
            self.trades = self.result.analyzers._TradeList.get_analysis()
        except Exception as e:
            print(f"Error extracting backtrader trades: {e}")
            
        return pd.DataFrame(self.trades)


class VectorbtAnalyzer(PerformanceAnalyzer):
    """Performance analyzer for VectorBT engine results."""
    
    def get_metrics(self):
        """Extract metrics from vectorbt portfolio."""
        metrics = {}
        
        if self.result is None:
            return metrics
        
        try:
            metrics = self.result.stats()
        
        except Exception as e:
            print(f"Error extracting vectorbt metrics: {e}")
            
        return metrics
    
    def generate_plot(self, plot_type="line", dynamic=False):
        """Generate plots for vectorbt results."""
        if self.result is None:
            return None
            
        try:
            port_ret = (self.result.asset_value() + self.result.cash()).pct_change()
            benchmark_ret = self.result.benchmark_returns()
            
            self.returns = pd.concat([port_ret, benchmark_ret], axis=1)
            self.returns.columns = ["Account", "Benchmark"]
            self.returns.fillna(0, inplace=True)
            perf_plot.create_pnl_plot(self.returns, self.config.date_rule, plot_type, dynamic)
        
        except Exception as e:
            print(f"Error plotting vectorbt results: {e}")
    
    def get_positions(self):
        """Get positions from vectorbt portfolio."""
        if self.result is None:
            return None
        
        self.positions = {}
        try:
            self.positions =  self.result.positions.records_readable
        except Exception as e:
            print(f"Error extracting vectorbt positions: {e}")
        
        return pd.DataFrame(self.positions)
    
    def get_trades(self):
        """Get trades from vectorbt portfolio."""
        if self.result is None:
            return None
        
        self.trades = {}
        try:
            self.trades = self.result.orders.records_readable
        except Exception as e:
            print(f"Error extracting vectorbt trades: {e}")
        
        return pd.DataFrame(self.trades)


def create_analyzer(engine: str, result=None, config=None):
    """Factory function to create the appropriate performance analyzer.
    
    Args:
        engine: Engine type ('backtrader', 'vectorbt')
        result: Result object from the backtest engine
        config: Configuration object containing engine settings
        
    Returns:
        Appropriate PerformanceAnalyzer subclass instance
        
    Raises:
        ValueError: If engine type is not supported
    """
    analyzers = {
        "backtrader": BacktraderAnalyzer,
        "vectorbt": VectorbtAnalyzer,
    }
    
    analyzer_class = analyzers.get(engine)
    if not analyzer_class:
        raise ValueError(f"Unsupported engine: {engine}. Supported engines: {list(analyzers.keys())}")
    
    return analyzer_class(result, config)