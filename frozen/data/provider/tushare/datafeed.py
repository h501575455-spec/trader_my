import datetime
import numpy as np
import pandas as pd
import tushare as ts
import time
from typing import Optional

from .. import TickerType
from ..base import DataFeed
from ...utils.util import rate_limiter, paginated_fetch
from ...utils.constants import DEFAULT_START, DEFAULT_END, TODAY
from ...utils.util import convert_datetime_columns
from ...utils.log import L

# Initialize logger
logger = L.get_logger("frozen")


class TushareDataFeed(DataFeed):
    """
    Tushare Data Feed.

    Use tushare api to create 6 different types of feed.
    - volume-price data
    - stock limit data
    - stock fundamental data
    - stock dividend data
    - stock suspend data
    - stock basic data
    """

    def __init__(self):

        from ...config import data_config
        datasource_cfg = data_config["datasource"]

        token = datasource_cfg["tushare"]["token"]
        ts.set_token(token)
        self.pro = ts.pro_api()

    @rate_limiter(max_calls_per_minute=500)
    def fetch_volume_price(self, ticker=None, start_date=None, end_date=None, asset="E", adj="", update=False):
        """Extract and transform instrument volume-price data"""
        start_date, end_date = self._get_period(start_date, end_date, update)
        query = ts.pro_bar(ts_code=ticker, start_date=start_date, end_date=end_date, asset=asset, adj=adj)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker", "vol": "volume"}, inplace=True)
        return query
    
    @rate_limiter(max_calls_per_minute=500)
    def fetch_stock_limit(self, ticker=None, start_date=None, end_date=None, update=False):
        """Extract and transform stock gain and loss limit data"""
        start_date, end_date = self._get_period(start_date, end_date, update)
        query = self.pro.stk_limit(ts_code=ticker, start_date=start_date, end_date=end_date)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    @rate_limiter(max_calls_per_minute=500)
    def fetch_stock_fundamental(self, ticker=None, start_date=None, end_date=None, update=False, **kwargs):
        """Extract and transform stock fundamental data"""
        start_date, end_date = self._get_period(start_date, end_date, update)
        default_fields = "ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv"
        fields = kwargs.get("fields", default_fields)
        query = self.pro.daily_basic(ts_code=ticker, start_date=start_date, end_date=end_date, fields=fields)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    @rate_limiter(max_calls_per_minute=500)
    def fetch_stock_dividend(self, ticker=None, update=False, **kwargs):
        """Extract and transform stock dividend data"""
        default_fields = "ts_code, end_date, ann_date, div_proc, stk_div, stk_bo_rate, stk_co_rate, cash_div, cash_div_tax, record_date, ex_date, pay_date, div_listdate, imp_ann_date, base_date, base_share"
        fields = kwargs.get("fields", default_fields)
        query = self.pro.dividend(ts_code=ticker, fields=fields)
        query = query[query["ann_date"]>=DEFAULT_START]  # only keep data after 20250101 due to A-share reform of listed companies
        query.drop_duplicates(subset=("ts_code", "end_date", "ann_date", "div_proc"), keep="first", inplace=True)
        if not self._check_validity(query):
            return 
        if update:
            cutoff_date = kwargs.get("cutoff", None)
            query = query[query["ann_date"]>=cutoff_date]
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    @rate_limiter(max_calls_per_minute=500)
    def fetch_stock_suspend(self, trade_date):
        """Extract and transform stock suspend status information"""
        query = self.pro.suspend_d(suspend_type="S", trade_date=trade_date)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    def fetch_stock_basic(self, **kwargs):
        """Extract and transform basic information about (de)listed stocks"""
        fields_str = "ts_code, symbol, name, area, industry, fullname, enname, cnspell, market, exchange, curr_type, list_status, list_date, delist_date, is_hs, act_name, act_ent_type"
        query = self.pro.stock_basic(fields=fields_str, **kwargs)
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    def _fetch_stock_fr_income(self, ticker=None, start_date=None, end_date=None, report_type=1):
        """Extract and transform stock financial report - income data"""
        query = self.pro.income(ts_code=ticker, start_date=start_date, end_date=end_date, report_type=report_type)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    def _fetch_stock_fr_balancesheet(self, ticker=None, start_date=None, end_date=None, report_type=1):
        """Extract and transform stock financial report - balance sheet data"""
        query = self.pro.balancesheet(ts_code=ticker, start_date=start_date, end_date=end_date, report_type=report_type)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    def _fetch_stock_fr_cashflow(self, ticker=None, start_date=None, end_date=None, report_type=1):
        """Extract and transform stock financial report - cash flow data"""
        query = self.pro.cashflow(ts_code=ticker, start_date=start_date, end_date=end_date, report_type=report_type)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    def fetch_stock_financial_report(self, ticker=None, start_date=None, end_date=None, fr_type="income"):
        merged = []
        for report_type in [1, 4, 5, 11]:
            if fr_type == "income":
                query = self._fetch_stock_fr_income(ticker, start_date, end_date, report_type=report_type)
            elif fr_type == "balancesheet":
                query = self._fetch_stock_fr_balancesheet(ticker, start_date, end_date, report_type=report_type)
            elif fr_type == "cashflow":
                query = self._fetch_stock_fr_cashflow(ticker, start_date, end_date, report_type=report_type)
            else:
                raise ValueError("Incorrect value of `fr_type`")
            merged.append(query)
        try:
            merged = pd.concat([df for df in merged if df is not None], axis=0)
            merged.sort_values(["end_date", "f_ann_date", "report_type", "update_flag"], ascending=[True, True, True, False], inplace=True)
            merged.drop_duplicates(subset=["end_date", "f_ann_date"], keep="first", inplace=True)
            merged.reset_index(drop=True, inplace=True)
            return merged
        except:
            return 
    
    def fetch_trade_calendar(self, exchange="", start_date=None, end_date=None):
        """Extract and transform exchange trade calendars"""
        query = self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        return query
    
    def fetch_industry_mapping(self, ticker=None, source="SW"):
        """Extract and transform industry category and tiering"""
        if source == "SW":
            query = self.pro.index_member_all(ts_code=ticker)
        elif source == "ZX":
            query = self.pro.ci_index_member(ts_code=ticker)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        query["source"] = source
        return query

    @rate_limiter(max_calls_per_minute=500)
    @paginated_fetch(max_records_per_call=6000, sleep_between_calls=0.2)
    def fetch_index_weight(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, update=False, **kwargs) -> Optional[pd.DataFrame]:
        """ 
        Notes:
        ------
        This method implements an intelligent pagination mechanism:
        1. If no date range is specified, fetches latest available data
        2. If date range is too large (>6000 records), automatically splits into chunks
        3. Uses exponential backoff for robust error handling
        4. Respects rate limits with configurable sleep intervals
        """
        
        # Get pagination parameters from decorator
        max_records_per_call = kwargs.get("max_records_per_call", 6000)
        sleep_between_calls = kwargs.get("sleep_between_calls", 0.2)
        
        # Handle default date parameters
        start_date, end_date = self._get_period(start_date, end_date, update)
        
        all_data = []
        current_end = end_date  # Start from the end date and work backwards
        attempt = 0
        max_attempts = 3
        
        logger.info(f"Fetching index weight data for {ticker} from {start_date} to {end_date}")
        
        while current_end >= start_date:
            attempt += 1
            try:
                # For the first call, try to get all data at once
                if current_end == end_date:
                    logger.info(f"Attempt {attempt}: Fetching data from {start_date} to {current_end}...")
                    query = self.pro.index_weight(
                        index_code=ticker,
                        start_date=start_date,
                        end_date=current_end
                    )
                    
                    if not self._check_validity(query):
                        logger.warning(f"No data returned for {ticker}")
                        return None
                    
                    # Check if we got the maximum number of records (indicating truncation)
                    if len(query) < max_records_per_call:
                        # We got all data in one call
                        logger.info(f"Retrieved {len(query)} records in single call")
                        query = convert_datetime_columns(query)
                        query.rename(columns={"ts_code": "ticker"}, inplace=True)
                        return query
                    else:
                        # Data was truncated, need pagination
                        logger.info(f"Data truncated at {max_records_per_call} records, implementing pagination...")
                        all_data.append(query)
                        
                        # Find the earliest trade_date in current batch for next pagination
                        # Since data comes in reverse chronological order, min date is the earliest we got
                        earliest_date = query["trade_date"].min()
                        current_end = self._get_previous_date(earliest_date)
                        
                        if current_end < start_date:
                            break
                            
                else:
                    # Subsequent pagination calls - work backwards from current_end
                    logger.info(f"Pagination call: Fetching data from {start_date} to {current_end}...")
                    query = self.pro.index_weight(
                        index_code=ticker,
                        start_date=start_date,
                        end_date=current_end
                    )
                    
                    if not self._check_validity(query) or len(query) == 0:
                        logger.info(f"Pagination complete - no more data before {current_end}")
                        break
                        
                    all_data.append(query)
                    logger.info(f"Retrieved {len(query)} additional records")
                    
                    # Check if we need more pagination
                    if len(query) < max_records_per_call:
                        logger.info(f"Pagination complete - last batch had {len(query)} records")
                        break
                    else:
                        # Continue pagination backwards
                        earliest_date = query["trade_date"].min()
                        current_end = self._get_previous_date(earliest_date)
                        
                        if current_end < start_date:
                            logger.info(f"Pagination complete - reached start date")
                            break
                
                # Reset attempt counter on success
                attempt = 0
                
                # Sleep between calls to respect rate limits
                if sleep_between_calls > 0:
                    time.sleep(sleep_between_calls)
                    
            except Exception as e:
                logger.error(f"Error in attempt {attempt}: {e}")
                
                if attempt >= max_attempts:
                    logger.error(f"Max attempts reached, giving up on current batch")
                    break
                    
                # Exponential backoff
                backoff_time = min(2 ** attempt, 10)
                logger.info(f"Backing off for {backoff_time} seconds...")
                time.sleep(backoff_time)
                continue
        
        # Combine all data
        if not all_data:
            logger.warning(f"No data retrieved for {ticker}")
            return None
            
        logger.info(f"Combining {len(all_data)} batches of data...")
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates that might occur at batch boundaries
        combined_data = combined_data.drop_duplicates(
            subset=["trade_date", "index_code", "con_code"], 
            keep="first"
        )
        
        # Sort by date and constituent code
        combined_data = combined_data.sort_values(["trade_date", "con_code"])
        combined_data = combined_data.reset_index(drop=True)
        
        # Apply standard transformations
        combined_data = convert_datetime_columns(combined_data)
        combined_data.rename(columns={"index_code": "ticker"}, inplace=True)
        
        total_records = len(combined_data)
        date_range = f"{combined_data['trade_date'].min()} to {combined_data['trade_date'].max()}"
        
        logger.info(f"Successfully retrieved {total_records} total records covering {date_range}")
        
        return combined_data
    
    def fetch_convertible_bond_basic(self):
        """Extract and transform basic information about convertible bonds"""
        fields_str = "ts_code, bond_full_name, bond_short_name, cb_code, stk_code, stk_short_name, maturity, par, issue_price, issue_size, remain_size, value_date, maturity_date, rate_type, coupon_rate, add_rate, pay_per_year, list_date, delist_date, exchange, conv_start_date, conv_end_date, conv_stop_date, first_conv_price, conv_price, rate_clause, put_clause, maturity_put_price, call_clause, reset_clause, conv_clause, guarantor, guarantee_type, issue_rating, newest_rating, rating_comp"
        query = self.pro.cb_basic(fields=fields_str)
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker"}, inplace=True)
        return query
    
    @rate_limiter(max_calls_per_minute=500)
    def fetch_convertible_bond_daily(self, ticker=None, start_date=None, end_date=None, update=False):
        """Extract and transform convertible bond bar data"""
        start_date, end_date = self._get_period(start_date, end_date, update, use_default=False)
        fields_str = "ts_code, trade_date, open, high, low, close, pre_close, change, pct_change, vol, amount, bond_value, bond_over_rate, cb_value, cb_over_rate"
        query = self.pro.cb_daily(ts_code=ticker, start_date=start_date, end_date=end_date, fields=fields_str)
        if not self._check_validity(query):
            return 
        query = convert_datetime_columns(query)
        query.rename(columns={"ts_code": "ticker", "vol": "volume"}, inplace=True)
        return query
    
    def _get_next_date(self, current_date, days_ahead: int = 1) -> str:
        """
        Get the next date for pagination.
        
        Parameters:
        -----------
        current_date: str or pandas.Timestamp
            Current date in YYYYMMDD format or pandas timestamp
        days_ahead: int
            Number of days to advance
            
        Returns:
        --------
        str
            Next date in YYYYMMDD format
        """
        if isinstance(current_date, str):
            date_obj = datetime.datetime.strptime(current_date, "%Y%m%d")
        else:
            # Handle pandas timestamp
            date_obj = pd.to_datetime(current_date)
            
        next_date = date_obj + datetime.timedelta(days=days_ahead)
        return next_date.strftime("%Y%m%d")
    
    def _get_previous_date(self, current_date, days_back: int = 1) -> str:
        """
        Get the previous date for backward pagination.
        
        Parameters:
        -----------
        current_date: str or pandas.Timestamp
            Current date in YYYYMMDD format or pandas timestamp
        days_back: int
            Number of days to go back
            
        Returns:
        --------
        str
            Previous date in YYYYMMDD format
        """
        if isinstance(current_date, str):
            date_obj = datetime.datetime.strptime(current_date, "%Y%m%d")
        else:
            # Handle pandas timestamp
            date_obj = pd.to_datetime(current_date)
            
        previous_date = date_obj - datetime.timedelta(days=days_back)
        return previous_date.strftime("%Y%m%d")

    def _check_validity(self, data):
        try:
            if data.empty:
                return False
        except:
            if data is None:
                return False
        else:
            return True
    
    def _get_period(self, start_date, end_date, update: bool, use_default: bool = True):
        if start_date is None:
            start_date = DEFAULT_START if use_default else None
        
        if end_date is None:
            end_date = DEFAULT_END if use_default else None
        
        if update:
            end_date = TODAY
        
        return start_date, end_date
    
    def get_instrument_list(
        self,
        ticker_type: TickerType = TickerType.LISTED_STOCK,
        code_only: bool = True,
        **kwargs
    ):
        """
        Get instrument list (stocks or indexes)
        
        Parameters:
        -----------
        ticker_type: TickerType
             Type of instrument to retrieve:
             - LISTED_STOCK: Listed stocks from specific market segments 
               (Main Board, ChiNext, STAR Market, etc. in SSE/SZSE)
             - LISTED_INDEX: Indexes published by exchanges or data providers
             - INDEX_COMPONENT: Constituent stocks of a specific index 
               (e.g., CSI 300 components)
        code_only: bool
            Whether to return only ticker codes, otherwise return full data
        **kwargs: other optional parameters
            For stock basic info: market, list_status, exchange, is_hs
            For index weights: index_code, trade_date, start_date, end_date
        """
        if ticker_type == TickerType.INDEX_COMPONENT:
            # Get index constituents
            index_code = kwargs.get("index_code", None)
            if index_code is None:
                raise ValueError("index_code is required when ticker_type is INDEX_COMPONENT")
            data = self.pro.index_weight(
                index_code=index_code,
                trade_date=kwargs.get("trade_date", None),
                start_date=kwargs.get("start_date", None),
                end_date=kwargs.get("end_date", None)
            )
        elif ticker_type == TickerType.LISTED_STOCK:
            # Get stock basic information from specific market segments
            data = self.pro.stock_basic(
                market=kwargs.get("market", ""),
                list_status=kwargs.get("list_status", "L"),
                exchange=kwargs.get("exchange", None),
                is_hs=kwargs.get("is_hs", None)
            )
        elif ticker_type == TickerType.LISTED_INDEX:
            # Get index basic information
            # NOTE: available market: MSCI(MSCI指数), CSI(中证指数), SSE(上交所指数), SZSE(深交所指数), CICC(中金指数), SW(申万指数), OTH(其他指数)
            data = self.pro.index_basic(
                market=kwargs.get("market", "主板")
            )
        elif ticker_type == TickerType.LISTED_CB:
            # Get convertible bond basic information
            data = self.pro.cb_basic(
                exchange=kwargs.get("exchange", "")
            )
        else:
            raise ValueError(f"Unsupported ticker_type: {ticker_type}")
        
        if code_only:
            try:
                data = data["ts_code"]
            except:
                data = data["con_code"]
            data = np.unique(data).tolist()
        
        return data