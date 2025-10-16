# -*- coding: utf-8 -*-
"""
Created on Mon Jan 9 16:34:21 2023

@author: lig
"""

import sys
import time
import random
import signal
import datetime
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Union, Tuple
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from itertools import zip_longest, islice

from ...utils import *
from ..riskmodel.optimizer import PortOpt
from ...factor.expression.base import Factor
from ...data.etl.dataload import DataLoadManager
from ...data.utils.constants import ONE_DAY

from ...basis import FrozenConfig

warnings.filterwarnings("ignore")


class FrozenBt(PortOpt, metaclass=ABCMeta):
    
    """
    The main engine of Frozen Backtest Framework.

    It is the base parent class used by the public `FactorFactory` class
    in `strategy` module. It should not be accessed directly by the user.

    Parameters
    ----------
    calling_file : str
        A string representation of the current working directory path.

    universe: tuple (optional)
        The instrument pool containing asset ticker in string format.
    
    _run_flag: bool
        The flag that indicates the running status of the engine, 
        i.e., assessing whether the strategy has been incorporated 
        into the framework and executed.

    _run_mode: str
        - "normal": The standard framework computation mode.
        - "parallel": The framework employs parallel computation, 
        applicable exclusively in scenarios involving multiple parameter 
        (tuning) circumstances.

    _num_cores: int
        The number of CPU cores used when conducting backtest.
        
    """
    
    call_counter = []    # cls variable, used to track all created instances of the FrozenBt class
    
    # Global flag to handle interruption
    _interrupted = False

    @classmethod 
    def _signal_handler(cls, signum, frame):
        """Global signal handler for KeyboardInterrupt"""
        print(f"\n🛑 Received interrupt signal. Stopping strategy execution...")
        cls._interrupted = True
        sys.exit(0)

    def __init__(self, calling_file: str, universe: tuple = None, config_set: dict = None) -> None:

        # Set up signal handler for Ctrl+C (only in main thread)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
        except ValueError:
            # signal.signal() only works in main thread
            # In parallel execution, this will fail but that's expected
            pass
        
        self.config = FrozenConfig(calling_file)
        if config_set:
            self.update_config(config_set)
        super().__init__(self.config)

        self.dataloader = DataLoadManager(self.config)
        
        if universe is not None:
            self.universe = universe
        else:
            self.univ()
            if not hasattr(self, "universe") or self.universe is None:
                raise ValueError("univ() method must set self.universe attribute")
        
        self._run_flag = False
        self._num_cores = mp.cpu_count()
        self._run_mode = "normal"
        self._calc_flag = False
        setattr(self.__class__, "_print_enabled", True)

        self.__dict__.update(self.config.get_all_params())
        
        # Add the instance to class variable call_counter during instantiation
        FrozenBt.call_counter.append(self)
    

    def __repr__(self):
        """String representation of a Framework object"""
        s = label_to_string("ENGINE TYPE", self.__class__.__base__.__name__)
        s += label_to_string("INDEX", self.index_code)
        s += label_to_string("BENCHMARK", self.benchmark)
        s += label_to_string("START DATE", self.start_date)
        s += label_to_string("END DATE", self.end_date)
        s += label_to_string("ASSET RANGE", self.asset_range)
        s += label_to_string("DATE RULE", self.date_rule)
        s += label_to_string("INIT CAPITAL", self.init_capital)
        s += label_to_string("SLIPPAGE RATE", self.slippage_rate)
        s += label_to_string("SLIPPAGE TYPE", self.slippage_type)
        s += label_to_string("COMMISSION", self.commission)
        s += label_to_string("STAMP DUTY", self.stamp_duty)
        s += label_to_string("TAKE PROFIT", self.take_profit)
        s += label_to_string("STOP LOSS", self.stop_loss)
        s += label_to_string("SELL RULE", self.sell_rule)
        return s
    

    def __str__(self):
        """Overloads `print` output of the object to view the expression of factor once executes calc() method."""
        if not hasattr(self, "alpha"):
            return self.__repr__()
        return self.alpha.__str__()


    @property
    def params(self):
        return self._params

    @property
    def runtime(self):
        """Total runtime of a strategy"""
        if not self._run_flag:
            print("Must run method run_backtest first!")
        else:
            t = self._time_end - self._time_start
            return f"Total runtime: {t:0.2f}s"

    @property
    def num_cores(self):
        return self._num_cores
    
    @num_cores.setter
    def num_cores(self, n: int):
        if isinstance(n, int):
            self._num_cores = n
        else:
            raise ValueError("Illegal param: parallel must be of type int.")

    @property
    def run_mode(self):
        return self._run_mode
    
    @run_mode.setter
    def run_mode(self, mode: str):
        if mode not in ["normal", "parallel"]:
            raise ValueError("Illegal param: run_mode must be `normal` or `parallel`.")
        else:
            self._run_mode = mode

    @classmethod
    def create_param_test(
            cls, 
            alpha: Factor, 
            batch_number: int, 
            varargs: dict = {}, 
            parallel: bool = True, 
            progress_bar: bool = True, 
            market_data: Tuple = None
        ) -> dict:
        """Initiate backtest cases and returns test params for parallel testing."""
        
        # NOTE: The cls() method calls FactorFactory, therefore no params should be given
        new_test = cls()
        new_test.alpha = alpha
        new_test.__dict__.update(varargs)
        new_test.enable_gif = False
        new_test._run_mode = "parallel" if parallel else "normal"
        setattr(new_test.__class__, "_print_enabled", False)
        details = {"batch": batch_number, "params": varargs}
        tqdm.write(f"Created a new test with date_rule of {new_test.date_rule} and asset_range of {new_test.asset_range}")
        tqdm.write(f"Batch {batch_number}: {details}")
        daily_return, ind = new_test.result_analysis(batch_number, progress_bar, market_data, plot=False, print_table=False)

        return details, daily_return, ind
    

    @classmethod
    def get_total_batch(cls):
        return len(cls.call_counter)

    @abstractmethod
    def univ(self) -> None:
        """
        Define instrument universe and set self.universe attribute.
        This method will be called automatically during initialization if universe is not provided.
        """
        self.universe = ...
        raise NotImplementedError("`universe` method must be implemented.")
        
    @abstractmethod
    def prepare_data(self) -> Union[Factor, Tuple]:
        raise NotImplementedError("`prepare_data` method must be implemented.")
    
    @abstractmethod
    @decorator.factor
    def calc(self, *args, **kwargs) -> Factor:
        raise NotImplementedError("`calc` method must be implemented.")
    
    
    @decorator.marketdata(attribute_name="_print_enabled")
    def getMarketData(self):
        """Call market data api from predefined data module."""
        try:
            benchmark = self.dataloader.load_volume_price("index_daily", "close", (self.benchmark,), start_date=self.start_date_extend, end_date=self.end_date, fillna=True)
            
            real_open, real_close, real_high, real_low = self.dataloader.load_volume_price("stock_daily_real", ("open", "close", "high", "low"), self.universe, start_date=self.start_date_extend, end_date=self.end_date, fillna=True)
            
            up_limit, down_limit = self.dataloader.load_volume_price("stock_daily_limit", ("up_limit", "down_limit"), self.universe, start_date=self.start_date_extend, end_date=self.end_date, fillna=True)
            
            hfq_close = self.dataloader.load_volume_price("stock_daily_hfq", "close", self.universe, start_date=self.start_date_extend, end_date=self.end_date, fillna=True)
            
            dividend = self.dataloader.load_stock_dividend("stock_dividend", self.universe)
            
            suspend = self.dataloader.load_stock_suspend("stock_suspend_status", start_date=self.start_date, end_date=self.end_date)
        
        except KeyboardInterrupt:
            raise KeyboardInterrupt("\n🛑 Market data loading interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error loading market data: {e}")
            raise

        mkt_data = (benchmark, real_open, real_close, real_high, real_low, up_limit, down_limit, hfq_close, dividend, suspend)

        return mkt_data


    @decorator.framework(attribute_name="_print_enabled")
    def FrozenEngine(
            self, 
            market_data: Tuple = None, 
            batch_number = None, 
            progress_bar: bool = True
        ):
        """
        The primary event-driven backtest framework cerebro.

        Parameters
        ----------
        batch_number: Union[None, int]
            The backtest batch number when running parameter tuning.
        
        progress_bar: bool
            The indicator to control whether to show progress bar.
        
        market_data: Tuple
            The market data needed during backtest.
        
        Returns
        ----------
        port_ret: pd.DataFrame
            The backtest result, including the strategy returns and 
            benchmark returns.
        
        order_history: pd.DataFrame
            The documentation of order history, including holdings 
            and portfolio wieghts.
        
        trade_records: pd.DataFrame
            The detailed trade records including all buy and sell transactions.
        """

        # Define constants
        timedelta = self.calendar.get_period_delta(self.date_rule)

        # Transform string fields to boolean
        # TODO
        sell_delay = True if self.sell_rule == "next-bar" else False

        # Create storage path for strategy results and initialize trading log
        L = FrozenLogger(self.config, self._run_mode)
        logger = L.get_logger()
        self.FULL_PATH = L.FULL_PATH
        
        # Create trade records DataFrame
        trade_records = pd.DataFrame(columns=[
            "trade_date", "instrument", "trade_direction", "num_shares", 
            "transaction_price", "transaction_amount", "transaction_cost"
        ])

        # Format factor data type
        if not isinstance(self.alpha, Factor):
            alpha_series = self.alpha[self.alpha.columns[0]]
            alpha_temp = alpha_series.swaplevel().unstack().sort_index()
            self.alpha = Factor(alpha_temp, alpha_series.name)
        
        # Apply factor preprocessing pipeline
        self.alpha = self._apply_factor_preprocessing(self.alpha)
        
        # Truncate factor data by config and deal with target asset range in percentage form
        score = self.alpha.data[self.start_date : self.end_date]
        total_instruments = len(score.columns)
        min_rank = round(self.asset_range[0] * total_instruments) if self.asset_range[0] < 1 else self.asset_range[0]
        max_rank = round(self.asset_range[1] * total_instruments) if self.asset_range[1] < 1 else self.asset_range[1]
        n_instruments = max_rank - min_rank
        if n_instruments < 1:
            raise FrozenError("No instruments available for backtest, please check the asset_range setting.")
        if n_instruments > len(score.columns):
            raise FrozenError("No enough instruments for selection, consider expanding the universe or relaxing the filter constraints.")    

        # Ensure minimum rank is not less than 0, and maximum rank does not exceed total instruments
        min_rank = max(0, min_rank)
        max_rank = min(total_instruments, max_rank)
        
        # Prepare market data for backtest
        benchmark, real_open, real_close, real_high, real_low, up_limit, down_limit, hfq_close, dividend, suspend = self.getMarketData() if market_data is None else market_data
        rets = hfq_close.pct_change()
        mkt_rets = benchmark.pct_change()

        # Resample data to the required frequency according to config
        # NOTE: Later, exclude the last row, as it is obtained from incomplete data
        period_score = score.resample(self.date_rule, label="right").mean()
        period_suspend = suspend.resample(self.date_rule, label="right")["ticker"]
        
        # Select target instruments for each period
        order_target = []
        # NOTE: Since period_suspend is an iterable object and if it has already been fully exhausted within the loop, 
        # it will be empty during the second iteration, causing the loop to terminate prematurely.
        # NOTE: Using zip() renders an incomplete order_target, while zip_longest() ensures the full iteration.
        # NOTE: Although screened_ticker is set to be the unscreened version if sp is None, it is still recommended that
        # suspend data is updated on a date_rule freqency basis.
        for i, (_, sp) in enumerate(zip_longest(range(len(period_score)), period_suspend, fillvalue=None)):
            try:
                score_na = period_score.iloc[i,:].dropna()
                ticker_by_order = period_score.iloc[i,:].sort_values(ascending=False).index
                # exclude stocks in suspension for the last period whilst retaining the ticker order
                if sp is not None:    
                    screened_ticker = list(filter(lambda tk: tk not in sp[1].tolist(), ticker_by_order))
                else:
                    screened_ticker = ticker_by_order
                order_target.append(screened_ticker[min_rank:max_rank] if not score_na.empty else order_target[i-1])
            except FrozenError as fe:
                print(f"Invalid alpha value! Make sure alpha on start date is not NA. Details: {fe}")
           
        # Initialize backtest key variables
        last_day_value = self.init_capital
        last_period_value = self.init_capital
        account_rets = []
        market_rets = []
        dates = []
        w = np.zeros((len(order_target), n_instruments))
        # NOTE: Exclude holidays where trades could not be executed when calculating the total number of trades
        total_order = 0
        win = 0
        
        # Configure progress bar
        disable = not progress_bar
        desc = "Performing Backtest" if self._run_mode == "normal" else f"Processing batch {batch_number}"
        pbar = tqdm(total=len(order_target), desc=desc, position=0, leave=True, disable=disable)

        if self.enable_gif:
            gif_plot = InteractivePlot()

        # Event-driven main loop
        try:
            for i in range(len(order_target)):
            
                target_instruments = order_target[i]
                calc_date = period_score.index[i]
                shift_date = self.calendar.adjust(calc_date, n_days=1)
                period_start = calc_date + ONE_DAY
                period_end = calc_date + timedelta if i != len(order_target) - 2 else self.end_date

                target_rets = rets[target_instruments]
                target_real_open = real_open[target_instruments]
                target_real_close = real_close[target_instruments]
                target_real_high = real_high[target_instruments]
                target_real_low = real_low[target_instruments]

                period_return = target_rets[period_start : period_end]
                period_real_open = target_real_open[period_start : period_end]
                period_real_close = target_real_close[period_start : period_end]
                period_real_high = target_real_high[period_start : period_end]
                period_real_low = target_real_low[period_start : period_end]

                pbar.update(1)

                # Deal with special cases

                # 1. Only one trade day in the entire period.
                #    - Since A-share is not T+0, we cannot buy and sell on the same day.
                if len(period_return) == 1:
                    logger.warning("Single trading day within the entire period, portfolio rebalancing will not be implemented.", extra={"trade_date": (calc_date + ONE_DAY).strftime("%Y-%m-%d")})
                    continue

                # 2. The next whole period is empty, which leads to two possible situations:
                #    - The next whole period is not trading day.
                #    - The next period is rebalancing period but rebalancing day has not yet arrived.
                if period_return.empty:
                    if i != len(order_target) - 1:
                        logger.warning("The whole period is holiday.", extra={"trade_date": (calc_date + ONE_DAY).strftime("%Y-%m-%d")})
                        continue
                    else:
                        logger.warning("The rebalancing day for the next cycle has not yet arrived, waiting for new position to open.", extra={"trade_date": (calc_date + ONE_DAY).strftime("%Y-%m-%d")})

                # 3. The backtest end_date is not the last period factor calc_date.
                if i == len(order_target) - 1 and calc_date > datetime.datetime.strptime(self.end_date, "%Y%m%d"):
                    tqdm.write("🔔 Warnings: Not consistent with date_rule for the last holding period.")
                    break

                # 4. There is holiday between factor calc_date and shift_date.
                if self.calendar.exist_holiday_between_dates(calc_date, shift_date):
                    logger.warning("Portfolio reallocation day is holiday, rebalancing deferred to next trading day.", extra={"trade_date": (calc_date + ONE_DAY).strftime("%Y-%m-%d")})

                # Apply risk model for portfolio optimization
                if n_instruments > 1:
                    # for covariance matrix estimation
                    in_sample, mkt_in_sample = None, None
                    if self.optimizer != "equal-weight":
                        target_rets, mkt_rets = target_rets.ffill().bfill(), mkt_rets.ffill().bfill()
                        in_sample = target_rets[calc_date - datetime.timedelta(days=self.cov_window) : calc_date]
                        mkt_in_sample = mkt_rets[calc_date - datetime.timedelta(days=self.cov_window) : calc_date]
                    # calculate portfolio weights
                    w[i] = self.calc_portfolio_weights(n=n_instruments, X=in_sample, X_mkt=mkt_in_sample)
                elif n_instruments == 1:
                    w[i] = 1.0
                else:
                    raise FrozenError("Available securities less than 1.")

                # 5. The last period factor data is complete, but future out of sample data is not yet available.
                #    - Output portfolio weight for future orders without backtesting.
                if i == len(order_target) - 1 and (calc_date == datetime.datetime.strptime(self.end_date,"%Y%m%d")
                    or shift_date > datetime.datetime.strptime(self.end_date,"%Y%m%d")):
                    tqdm.write("🔔 Warnings: Detected empty or incomplete dataframe for the last holding period.")
                    break
                
                # Initialize middle variables
                num_share_list = [0] * n_instruments
                hold_flag = [False] * n_instruments
                loss_lim_flag = [False] * n_instruments
                profit_lim_flag = [False] * n_instruments
                avail_cash = last_period_value
                start_cost = np.zeros(n_instruments)
                end_recover = np.zeros(n_instruments)

                # Instrument holding period loop
                # Iterate through period trade days
                for date in period_return.index:

                    daily_account_value = 0
                    end_total_position = 0
                    buy_flag = [False] * n_instruments
                    sell_flag = [False] * n_instruments

                    # Iterate across target instruments
                    for k, instrument in enumerate(target_instruments):

                        # Set sell flag signal
                        if date == period_return.index[-1] and num_share_list[k] > 0:
                            sell_flag[k] = True
                        
                        if (not hold_flag[k]) and (not (loss_lim_flag[k] or profit_lim_flag[k])):
                            # Check instrument buy status
                            buy_price = period_real_open.loc[date, instrument]
                            buy_status, suspend_status, limit_status = execute.inst_trade_status(instrument, date, buy_price, up_limit, down_limit, suspend, direction=TransactionType.BUY)
                            if buy_status and date != period_return.index[-1]:
                                # Decide the number of shares to hold
                                # - The simple way
                                # num_shares = int((last_period_value * w[i][k] / buy_price) / self.trade_unit)
                                # num_shares = 0 if num_shares < 1 else num_shares  # the characteristic of A share, if num_shares < 1, dump the stock
                                # num_shares *= self.trade_unit
                                # - The accurate way
                                init_cash = last_period_value * w[i][k]
                                num_shares = solver.share_solver(init_cash, buy_price, self.slippage_rate, self.slippage_type, self.commission, self.min_cost, self.trade_unit)
                                num_share_list[k] = num_shares
                                # Apply slippage
                                buy_price_with_slippage = execute.apply_slippage(buy_price, self.slippage_rate, self.slippage_type, transaction_type=TransactionType.BUY)
                                inst_begin_position = num_shares * buy_price_with_slippage
                                # Calculate commission
                                buy_commission = inst_begin_position * self.commission
                                buy_commission = buy_commission if buy_commission >= self.min_cost else self.min_cost
                                buy_commission = buy_commission if num_shares > 0 else 0
                                # Calculate account available cash
                                buy_cost = inst_begin_position + buy_commission
                                if avail_cash < buy_cost:
                                    extra_cash = abs(avail_cash - buy_cost)
                                    logger.warning(
                                        f"Account available cash is not enough for trading target instrument {instrument}. Extra cash needed: {extra_cash:.2f}.",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                    logger.info(
                                        f"Filling up account available cash with {extra_cash:.2f}.",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                    avail_cash += extra_cash
                                avail_cash = max(avail_cash - buy_cost, 0)
                                # Log trade info
                                logger.info(
                                    f"Placing order (long instrument) on {instrument} for {num_shares} shares at price {buy_price:.2f}, with transcation cost of {buy_commission:.2f}. Available cash: {avail_cash:.2f}",
                                    extra={"trade_date": date.strftime("%Y-%m-%d")}
                                )
                                
                                # Add trade record to DataFrame
                                trade_records = pd.concat([trade_records, pd.DataFrame([{
                                    "trade_date": date,
                                    "instrument": instrument,
                                    "trade_direction": "buy",
                                    "num_shares": num_shares,
                                    "transaction_price": buy_price_with_slippage,
                                    "transaction_amount": inst_begin_position,
                                    "transaction_cost": buy_commission
                                }])], ignore_index=True)
                                
                                # Set flag indicators
                                hold_flag[k], buy_flag[k] = True, True
                                # Store buy cost
                                start_cost[k] = inst_begin_position + buy_commission
                            else:
                                num_shares = 0
                                num_share_list[k] = num_shares
                                if suspend_status:
                                    logger.warning(
                                        f"Due to suspension events, cannot place orders on {instrument}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                if limit_status:
                                    logger.warning(
                                        f"Instrument open price reached up limit, cannot place orders on {instrument}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                if date == period_return.index[-1]:
                                    logger.warning(
                                        f"Last day of trading period, cannot place orders on {instrument}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                        
                        if buy_flag[k]:
                            dayend_price = period_real_close.loc[date, instrument]
                            dayend_position = num_share_list[k] * dayend_price
                            daily_account_value += dayend_position
                            continue
                        
                        # For the instruments being held, perform the following steps
                        if hold_flag[k]:
                            # 1. Check dividend event
                            if instrument in dividend.keys():
                                info = dividend[instrument]
                                if date in info["ex_date"].tolist():
                                    logger.info(
                                        f"Dividend event occurs for {instrument} on {date}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                    # NOTE: Sometimes there exists multiple indices for a single date.
                                    # NOTE: First cash dividend, then share dividend.
                                    cash_div = np.array(info[info["ex_date"]==date]["cash_div"])[0] * num_share_list[k]
                                    stk_div = int(num_share_list[k] * (1 + np.array(info[info["ex_date"]==date]["stk_div"])[0]))
                                    avail_cash += cash_div
                                    num_share_list[k] = stk_div
                                    logger.info(
                                        f"Rebalancing account holdings for {instrument}: {num_share_list[k]} shares.",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                            
                            # 2. Monitor stop profit & loss point
                            # NOTE: Limitation: cannot monitor instrument intraday price trend
                            day_high_price = period_real_high.loc[date, instrument]
                            day_low_price = period_real_low.loc[date, instrument]
                            day_open_price = period_real_open.loc[date, instrument]
                            # Determine which condition to check based on the dice roll
                            random.seed(self.seed)
                            dice = random.random()
                            # Calculate stop-loss and stop-profit thresholds
                            stop_loss_threshold = day_low_price / day_open_price - 1
                            stop_profit_threshold = day_high_price / day_open_price - 1
                            # Arbitrary threshold to control which condition to check first
                            if dice < 0.5:
                                if stop_loss_threshold <= self.stop_loss:
                                    sell_flag[k], loss_lim_flag[k], hold_flag[k] = True, True, False
                                    logger.warning(
                                        f"Stop-loss point occurs for {instrument} on {date}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                elif stop_profit_threshold >= self.take_profit:
                                    sell_flag[k], profit_lim_flag[k], hold_flag[k] = True, True, False
                                    logger.warning(
                                        f"Stop-profit point occurs for {instrument} on {date}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                            else:
                                if stop_profit_threshold >= self.take_profit:
                                    sell_flag[k], profit_lim_flag[k], hold_flag[k] = True, True, False
                                    logger.warning(
                                        f"Stop-profit point occurs for {instrument} on {date}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                elif stop_loss_threshold <= self.stop_loss:
                                    sell_flag[k], loss_lim_flag[k], hold_flag[k] = True, True, False
                                    logger.warning(
                                        f"Stop-loss point occurs for {instrument} on {date}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                        else:
                            continue

                        if sell_flag[k]:
                            if profit_lim_flag[k]:
                                sell_price = round(day_open_price * (1 + self.take_profit), 2)
                            elif loss_lim_flag[k]:
                                sell_price = round(day_open_price * (1 + self.stop_loss), 2)
                            else:
                                sell_price = period_real_close.loc[date, instrument]
                            # Check instrument sell status
                            sell_status, suspend_status, limit_status = execute.inst_trade_status(instrument, date, sell_price, up_limit, down_limit, suspend, direction=TransactionType.SELL)
                            if sell_status:
                                # Apply slippage
                                sell_price_with_slippage = execute.apply_slippage(sell_price, self.slippage_rate, self.slippage_type, transaction_type=TransactionType.SELL)
                                inst_end_position = num_share_list[k] * sell_price_with_slippage
                                # Deal with sell commission
                                sell_commission = inst_end_position * self.commission
                                sell_commission = sell_commission if sell_commission >= self.min_cost else self.min_cost
                                # Deal with sell tax
                                sell_tax = inst_end_position * self.stamp_duty
                                sell_transaction_cost = sell_commission + sell_tax
                                logger.info(
                                    f"Sell (short) instrument {instrument} at price {sell_price:.2f} for {inst_end_position:.2f}. Transaction cost: {sell_transaction_cost:.2f}",
                                    extra={"trade_date": date.strftime("%Y-%m-%d")}
                                )
                                
                                # Add trade record to DataFrame
                                trade_records = pd.concat([trade_records, pd.DataFrame([{
                                    "trade_date": date,
                                    "instrument": instrument,
                                    "trade_direction": "sell",
                                    "num_shares": num_share_list[k],
                                    "transaction_price": sell_price_with_slippage,
                                    "transaction_amount": inst_end_position,
                                    "transaction_cost": sell_transaction_cost
                                }])], ignore_index=True)
                                
                                # Calculate available cash
                                avail_cash = avail_cash + inst_end_position - sell_transaction_cost
                                num_share_list[k] = 0
                                sell_flag[k] = False
                                # Store sell value
                                end_recover[k] = inst_end_position - sell_transaction_cost
                            else:
                                # Assume instruments can be sold
                                inst_end_position = num_share_list[k] * sell_price
                                avail_cash += inst_end_position
                                num_share_list[k] = 0
                                if suspend_status:
                                    logger.warning(
                                        f"Due to suspension events, cannot sell {instrument}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                if limit_status:
                                    logger.warning(
                                        f"Instrument open price reached down limit, cannot sell {instrument}",
                                        extra={"trade_date": date.strftime("%Y-%m-%d")}
                                    )
                                logger.info(
                                    f"Sell (short) virtual instrument {instrument} at previous price {sell_price:.2f} for {inst_end_position:.2f}. Transaction cost: {sell_transaction_cost:.2f}",
                                    extra={"trade_date": date.strftime("%Y-%m-%d")}
                                )
                                # Store assumed sell value
                                end_recover[k] = inst_end_position

                        dayend_price = period_real_close.loc[date, instrument]
                        dayend_position = num_share_list[k] * dayend_price
                        daily_account_value += dayend_position

                    # Report available cash
                    if date == shift_date:
                        logger.info(f"Account available cash after shift day: {avail_cash:.2f}", extra={"trade_date": date.strftime("%Y-%m-%d")})

                    daily_account_value += avail_cash
                    daily_pnl = daily_account_value / last_day_value - 1
                    account_rets.append(daily_pnl)
                    market_rets.append(mkt_rets.squeeze()[date])
                    last_day_value = daily_account_value
                    dates.append(date)

                    if self.enable_gif:
                        gif_plot.update(dates, account_rets, market_rets, is_first_update=not i)

                period_account_value = end_total_position + avail_cash
                period_pnl = period_account_value - last_period_value
                last_period_value = period_account_value
                logger.info(f"Account period PnL: {period_pnl:.2f}", extra={"trade_date": date.strftime("%Y-%m-%d")})
                logger.info(f"Account value: {period_account_value:.2f}", extra={"trade_date": date.strftime("%Y-%m-%d")})
                
                win += (end_recover > start_cost).sum()
                total_order += n_instruments
        
        except KeyboardInterrupt:
            pbar.close()
            raise KeyboardInterrupt("\n🛑 Backtest execution interrupted by user")
        
        pbar.close()

        if self.enable_gif:
            gif_plot.show()

        total_pnl = period_account_value - self.init_capital
        logger.info(f"Account total PnL: {total_pnl:.2f}", extra={"trade_date": date.strftime("%Y-%m-%d")})
        
        mkt_rets = mkt_rets.loc[dates].values.squeeze()
        port_ret = pd.DataFrame({"Account": account_rets, "Benchmark": mkt_rets}, index = dates)
        self.win_rate = win / total_order

        # Retrieve order information
        order_history = pd.DataFrame()
        adj_date = self.calendar.next_trade_dates(period_score.index)
        for order, weight, date in islice(zip(order_target, w, adj_date), len(order_target)):
            # Exclude situations where weights are zeros, could happen for two reasons:
            # - The whole portfolio repositioning period is holiday.
            # - The scheduled portfolio reallocation time has not yet arrived.
            if not np.any(weight):
                continue
            period_order = pd.DataFrame()
            period_order["trade_date"] = [date] * len(order)
            period_order["instrument"] = order
            period_order["weight"] = weight
            order_history = pd.concat([order_history, period_order])
        
        order_history.set_index(["trade_date", "instrument"], inplace=True)
    
        return port_ret, order_history, trade_records
    

    @decorator.result(attribute_name="_print_enabled")
    def result_analysis(
            self, 
            batch_number: int = None, 
            progress_bar: bool = True, 
            market_data: Tuple = None, 
            plot: bool = True, 
            plot_type: str = None, 
            dynamic: bool = False, 
            print_table: bool = True, 
            positions: bool = False, 
            trades: bool = False, 
            html: bool = False, 
            send_backend = False, 
        ):
        """
        Analyse portfolio performance, generate plots and major 
        assessment indicators of a strategy, and controls the 
        output of html and excel format.
        """

        port_ret, order_history, trade_records = self.FrozenEngine(market_data, batch_number, progress_bar)
        
        # Metrics table
        table = Report.print_evaluation_table(port_ret, self.win_rate, print_table)

        # Positions xslx sheet
        if positions:
            Report.to_excel(order_history, self.FULL_PATH, "positions")
        
        # Trades xslx sheet
        if trades:
            Report.to_excel(trade_records, self.FULL_PATH, "trades")

        # Performance web page
        if html:
            Report.to_quantstats(port_ret, self.FULL_PATH)
        
        # Performance plot
        if plot:
            perf_plot.create_pnl_plot(port_ret, self.date_rule, plot_type, dynamic=dynamic)
        
        self._time_end = time.perf_counter()
        self._run_flag = True

        ind = pd.DataFrame(table)

        if send_backend:
            return port_ret, ind, order_history, trade_records
        else:
            return port_ret, ind
    

    def run_backtest(
            self, 
            progress_bar: bool = True, 
            plot: bool = True, 
            plot_type: str = "line", 
            dynamic: bool = False, 
            print_table: bool = True, 
            positions: bool = False, 
            trades: bool = False, 
            html: bool = False, 
            send_backend = False,
            market_data = None
        ):
        
        self._time_start = time.perf_counter()

        # Check factor calculation status
        self._check_calc_flag()

        if self.alpha is None:
            raise FrozenError("Factor is not returned in calc() method")
        
        engine_handlers = {
            "backtrader": self._handle_backtrader_engine,
            "vectorbt": self._handle_vectorbt_engine,
            "frozen": self._handle_frozen_engine
        }
        
        handler = engine_handlers.get(self.engine)
        if not handler:
            raise ValueError(f"Unsupported engine: {self.engine}")
        
        return handler(
            progress_bar=progress_bar,
            plot=plot,
            plot_type=plot_type,
            dynamic=dynamic,
            print_table=print_table,
            positions=positions,
            trades=trades,
            html=html,
            send_backend=send_backend,
            market_data=market_data
        )
    
    def _handle_backtrader_engine(self, **kwargs):
        """Backtrader engine handler"""
        from .utils import Frozen2Backtrader, BacktraderAnalyzer
        
        bt_strategy = Frozen2Backtrader(self, mode="factor_selection")
        pf = bt_strategy.run()
        
        analyzer = BacktraderAnalyzer(pf, self.config)
        results = analyzer.analyze(
            plot=kwargs.get("plot", True),
            plot_type=kwargs.get("plot_type", "line"),
            dynamic=kwargs.get("dynamic", False),
            print_table=kwargs.get("print_table", True),
            positions=kwargs.get("positions", False),
            trades=kwargs.get("trades", False),
            html=kwargs.get("html", False),
            send_backend=kwargs.get("send_backend", False)
        )
        
        return results
    
    def _handle_vectorbt_engine(self, **kwargs):
        """VectorBT engine handler"""
        from .utils import Frozen2VectorBT, VectorbtAnalyzer
        
        vbt_strategy = Frozen2VectorBT(self, mode="factor_selection")
        pf = vbt_strategy.run_backtest()
        
        analyzer = VectorbtAnalyzer(pf, self.config)
        results = analyzer.analyze(
            plot=kwargs.get("plot", True),
            plot_type=kwargs.get("plot_type", "line"),
            dynamic=kwargs.get("dynamic", False),
            print_table=kwargs.get("print_table", True),
            positions=kwargs.get("positions", False),
            trades=kwargs.get("trades", False),
            html=kwargs.get("html", False),
            send_backend=kwargs.get("send_backend", False)
        )
        
        return results
    
    def _handle_frozen_engine(self, **kwargs):
        """Frozen engine handler"""
        
        results = self.result_analysis(
            progress_bar=kwargs.get("progress_bar", True),
            plot=kwargs.get("plot", True),
            plot_type=kwargs.get("plot_type", "line"),
            dynamic=kwargs.get("dynamic", False),
            print_table=kwargs.get("print_table", True),
            positions=kwargs.get("positions", False),
            trades=kwargs.get("trades", False),
            html=kwargs.get("html", False),
            send_backend=kwargs.get("send_backend", False),
            market_data=kwargs.get("market_data", None)
        )
        
        return ({"returns": results[0], "metrics": results[1], 
                "positions": results[2], "trades": results[3]} 
                if kwargs.get("send_backend") else results[0])
    

    def param_tuning(
            self, 
            params: list, 
            parallel: bool = True, 
            backend: str = None, 
            progress_bar: bool = True
        ):

        assert (parallel and (self._run_mode == "parallel")) or \
               (not parallel and self._run_mode == "normal"), \
               "Parameter not match: `run_mode` and `parallel` must match."
        
        # Check factor calculation status
        self._check_calc_flag()

        if parallel:
            assert backend is not None, "Please indicate a backend for parallel processing. Supoorted backends are: `loky`, `multiprocessing` and `threading`."
            # Prepare market data for parallel running
            market_data = self.getMarketData()
            with tqdm_joblib(desc="Parallel Processing", total=len(params), position=0, leave=True) as pbar:
                # NOTE: Supported backends:
                # - "loky": used by default, can induce some communication and memory overhead when exchanging input and output data with the worker Python processes.
                # - "multiprocessing": process-based backend based on `multiprocessing.Pool`. Less robust than `loky`.
                # - "threading": a very low-overhead backend but it suffers from the Python GIL if the called function relies a lot on Python objects.
                # NOTE: it appears that lru_cache fails under multiprocessing mode, perhaps we should resort 
                # to shared cache mechanism for multiprocessing or setting maxsize=None for multithreading
                # NOTE: if we don't use parallel_backend, there will be a problem concerning the update of call_counter, managed to solve by adding prefer="threads" in Parallel
                results = Parallel(n_jobs=self._num_cores, backend=backend, verbose=0, prefer="threads")(
                            delayed(self.create_param_test)(self.alpha, 
                                                            batch_number=i+1, 
                                                            varargs=param, 
                                                            parallel=parallel, 
                                                            progress_bar=progress_bar, 
                                                            market_data=market_data) 
                            for i, param in enumerate(params))
        else:
            results = []
            for i, param in enumerate(params):
                details, port_ret, ind = self.create_param_test(self.alpha, 
                                                                batch_number=i+1, 
                                                                varargs=param, 
                                                                parallel=parallel, 
                                                                progress_bar=progress_bar)
                print(f"Result for batch {i+1} with {details}: {ind}")
                results.append((details, port_ret, ind))
        
        port_rets, inds = Report.report_batch_results(results, parallel=parallel)

        return port_rets, inds


    def run_batch(
            self, 
            progress_bar: bool = True, 
            plot_type: str = None
        ):
        """
        Analyse portfolio performance, generate plots and major 
        assessment indicators of a strategy, and controls the 
        output of html and excel format.
        """

        market_data = self.getMarketData()
        self._check_calc_flag()
        if self.alpha.shape[1] > 1:
            alpha = self.alpha.copy()
            metrics_table = pd.DataFrame()
            for i, col in enumerate(alpha.columns):
                self.alpha = pd.DataFrame(alpha[col])
                port_ret, _, _ = self.FrozenEngine(market_data, batch_number=i+1, progress_bar=progress_bar)
                table = Report.print_evaluation_table(port_ret, self.win_rate, print_table=False)
                perf_plot.create_pnl_plot(port_ret, self.date_rule, plot_type)
                metrics_table = pd.concat([metrics_table, pd.DataFrame(table, index=[col])], axis=0)
        
        self._run_flag = True

        return metrics_table


    def _apply_factor_preprocessing(self, factor: Factor) -> Factor:
        """Apply factor preprocessing pipeline based on configuration"""
        from ...factor.utils.preprocess import normalize, standardize, clip_mad, winsorize_quantile, industry_neutralize
        from ...data.database import DatabaseTypes
        from ...utils.calendar import CalendarTypes
        
        processed_factor = factor
        
        # Skip preprocessing if auto_apply is False and no specific methods are enabled
        if not self.preprocess_auto_apply and not any([
            self.normalize_enabled and self.normalize_auto_apply,
            self.standardize_enabled and self.standardize_auto_apply,
            self.clip_enabled and self.clip_auto_apply,
            self.winsorize_enabled and self.winsorize_auto_apply,
            self.industry_neutralize_enabled and self.industry_neutralize_auto_apply
        ]):
            return processed_factor
        
        # Apply preprocessing methods according to the configured order
        for method in self.preprocess_order:
            
            if method == "normalize" and self.normalize_enabled and (self.preprocess_auto_apply or self.normalize_auto_apply):
                processed_factor = normalize(
                    processed_factor,
                    cross_section=self.normalize_cross_section,
                    expanding=self.normalize_expanding,
                    window=self.normalize_window if not self.normalize_expanding else None
                )
            
            elif method == "standardize" and self.standardize_enabled and (self.preprocess_auto_apply or self.standardize_auto_apply):
                processed_factor = standardize(
                    processed_factor,
                    cross_section=self.standardize_cross_section,
                    expanding=self.standardize_expanding,
                    window=self.standardize_window if not self.standardize_expanding else None
                )
            
            elif method == "clip" and self.clip_enabled and (self.preprocess_auto_apply or self.clip_auto_apply):
                processed_factor = clip_mad(
                    processed_factor,
                    expanding=self.clip_expanding,
                    window=self.clip_window if not self.clip_expanding else None,
                    multiplier=self.clip_multiplier
                )
            
            elif method == "winsorize" and self.winsorize_enabled and (self.preprocess_auto_apply or self.winsorize_auto_apply):
                processed_factor = winsorize_quantile(
                    processed_factor,
                    limits=tuple(self.winsorize_limits),
                    expanding=self.winsorize_expanding,
                    window=self.winsorize_window if not self.winsorize_expanding else None
                )
            
            elif method == "industry_neutralize" and self.industry_neutralize_enabled and (self.preprocess_auto_apply or self.industry_neutralize_auto_apply):
                # Get industry mapping for neutralization
                from ...utils.industry import get_industry_manager

                industry_manager = get_industry_manager(database_type=DatabaseTypes(self.database))
                industry_mapping = industry_manager.get_industry_time_series(
                    self.universe,
                    start_date=self.start_date_lookback,
                    end_date=self.end_date,
                    classification=self.industry_neutralize_classification,
                    calendar_type=CalendarTypes.NONE
                )
                
                processed_factor = industry_neutralize(
                    processed_factor,
                    industry_mapping=industry_mapping,
                    method=self.industry_neutralize_method,
                )
        
        return processed_factor


    def update_config(self, config: dict = None, **kwargs):
        """
        Update configuration values with proper synchronization 
        between FactorFactory and FrozenConfig.
        
        Parameters:
        -----------
        config : dict, optional
            Dictionary of configuration key-value pairs
        **kwargs : dict
            Configuration updates as keyword arguments
            
        Examples:
        ---------
        strategy.update_config({'preprocess_auto_apply': True, 'start_date': '20230101'})
        strategy.update_config(preprocess_auto_apply=True, start_date='20230101')
        """
        # Combine config dict and kwargs
        updates = {}
        if config is not None and isinstance(config, dict):
            updates.update(config)
        if kwargs:
            updates.update(kwargs)
        
        if not updates:
            return
            
        # Update strategy object attributes
        self.__dict__.update(updates)
        
        # Synchronize to underlying FrozenConfig object
        for k, v in updates.items():
            self.config.update_config(k, v)


    def sync_from_config(self):
        """
        Synchronize strategy attributes from the underlying 
        FrozenConfig object.
        
        This method should be called after making changes 
        directly to self.config to ensure the strategy object 
        reflects the latest configuration.
        
        Examples:
        ---------
        strategy.config.preprocess_auto_apply = True
        strategy.sync_from_config()  # Sync changes to strategy
        """
        updated_params = self.config.get_all_params()
        self.__dict__.update(updated_params)


    def _check_calc_flag(self):
        """Check factor calculation status"""
        if not self._calc_flag:
            self.alpha = self.calc()
            self._calc_flag = True