import warnings
import pandas as pd
from tqdm import tqdm

from ..utils.log import L
from ..provider import ProviderTypes
from ..provider.factory import ProviderFactory
from ..database import DatabaseTypes
from ..database.factory import DatabaseFactory
from ...utils.calendar import Calendar, CalendarTypes
from ..utils.constants import TODAY, DEFAULT_START
from .dataload import DataLoadManager

warnings.filterwarnings("ignore")

logger = L.get_logger("frozen")


class DataFeedManager:
    """
    The Data (E)xtract, (T)ransform, (L)oad Pipeline Module.
    Implement a robust ETL pipeline for efficient data processing 
    and storage.
   
   The pipeline consists of three primary stages:
    - Step 1: Extract
    Retrieve raw data from specified API, ensuring comprehensive data 
    acquisition.

    - Step 2: Transform
    Processes and refines the extracted data, applying necessary 
    cleansing, normalization, and structuring operations to prepare 
    it for analysis and storage.

    - Step 3: Load
    Dump the transformed data into target database, optimizing for 
    data integrity and query performance.

    This modular approach ensures scalability, maintainability, and 
    flexibility in handling diverse data sources and formats. The 
    pipeline is designed to accommodate potential future expansions 
    and modifications to meet evolving data management requirements.
    """

    def __init__(
        self,
        provider_type: ProviderTypes,
        database_type: DatabaseTypes,
        calendar_type: CalendarTypes
    ):
        self.provider = ProviderFactory.create_data_feed(provider_type)
        self.handler = DatabaseFactory.create_database_connection(database_type)
        self.handler.init_db()
        self.calendar = Calendar(calendar_type)
        self.dataloader = DataLoadManager(database_type)

    def fetch_volume_price_data(self, table_name, ticker_list=None, start_date=None, end_date=None, asset="E", adj="", update=False):
        logger.info(f"Begin to fetch {table_name} data, ticker number: {len(ticker_list) if ticker_list else 0}, update={update}")
        # Check if table exists, create table if not exists
        if not self.handler._check_table_exists(table_name):
            self.handler.create_volume_price_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} volume-price data.")
        
        # Incremental update based on existing data
        if update:
            logger.info(f"Execute incremental update mode for {table_name}")
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            logger.info(f"Found {len(table_ticker_list)} tickers in the database")

            # Step 1: Forward update (update latest data)
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremental fetch:"):
                start_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_volume_price(ticker=ticker, start_date=start_date, end_date=end_date, asset=asset, adj=adj, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            # Step 2: Backward fill (supplement historical data)
            logger.info(f"Checking for historical data gaps for {table_name}")
            try:
                # Load stock basic info to get listing dates
                stock_basic = self.dataloader.load_basic_info("stock_basic")
                stock_basic_dict = stock_basic[stock_basic["list_status"] == "L"].set_index("ticker")[["list_date"]].to_dict()["list_date"]

                backfill_count = 0
                for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Backfill check:"):
                    # Get listing date for this ticker
                    if ticker not in stock_basic_dict:
                        continue
                    
                    list_date = stock_basic_dict[ticker]
                    if isinstance(list_date, pd.Timestamp):
                        list_date = list_date.strftime("%Y%m%d")
                    
                    # Get earliest date in database for this ticker
                    earliest_date = self.handler._get_earliest_ticker_date(table_name, ticker)
                    if earliest_date is None:
                        continue

                    # Calculate the actual start date (max of list_date and DEFAULT_START)
                    actual_start = max(list_date, DEFAULT_START)
                    if not self.calendar.is_trade_day(actual_start):
                        actual_start = self.calendar.adjust(actual_start, n_days=1)
                        actual_start = actual_start.strftime("%Y%m%d")
                    
                    # Check if backfill is needed
                    if earliest_date > actual_start:
                        # Need to backfill: from actual_start to day before earliest_date
                        backfill_end = self.calendar.adjust(earliest_date, n_days=-1)
                        if isinstance(backfill_end, pd.Timestamp):
                            backfill_end = backfill_end.strftime("%Y%m%d")
                        logger.info(f"Backfilling {ticker}: {actual_start} to {backfill_end}")
                        
                        data = self.provider.fetch_volume_price(ticker=ticker, start_date=actual_start, end_date=backfill_end, asset=asset, adj=adj, update=False)
                        if not self._check_validity(data):
                            logger.warning(f"Table {table_name}, {ticker} backfill data is empty.")
                            continue
                        self.handler._insert_df_to_table(data, table_name)
                        backfill_count += 1
                
                logger.info(f"Backfilled historical data for {backfill_count} tickers")
            except Exception as e:
                logger.warning(f"Could not perform historical backfill: {e}")
            
            # for tickers other than existing ticker
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_volume_price(ticker=ticker, start_date=start_date, end_date=end_date, asset=asset, adj=adj, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            logger.info(f"Execute normal insert mode for {table_name}")
            # Perform data extraction, transformation and insertion in a loop
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_volume_price(ticker=ticker, start_date=start_date, end_date=end_date, asset=asset, adj=adj)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    logger.debug(f"Skip {ticker}, data already exists")
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_limit_data(self, table_name="stock_daily_limit", ticker_list=None, start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_limit_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} price data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremeantal fetch:"):
                start_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_stock_limit(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_stock_limit(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_limit(ticker=ticker, start_date=start_date, end_date=end_date)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_fundamental_data(self, table_name="stock_daily_fundamental", ticker_list=None, start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_fundamental_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremeantal fetch:"):
                start_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_stock_fundamental(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_stock_fundamental(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_fundamental(ticker=ticker, start_date=start_date, end_date=end_date)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_dividend_data(self, table_name="stock_dividend", ticker_list=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_dividend_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremeantal fetch:"):
                next_ticker_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_stock_dividend(ticker=ticker, update=True, cutoff=next_ticker_date)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_stock_dividend(ticker=ticker)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_dividend(ticker=ticker)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_suspend_data(self, table_name="stock_suspend_status", start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_suspend_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            table_date = self.handler._get_table_date(table_name, latest=True)
            start_date = self.calendar.adjust(table_date, n_days=1)
            tradeday_list = self.calendar.generate(start_date, TODAY).strftime("%Y%m%d")
            date_for_query = self.calendar.generate(start_date, TODAY).strftime("%Y-%m-%d")
            for trade_date, format_date in tqdm(zip(tradeday_list, date_for_query), total=len(tradeday_list), desc="Incremental fetch"):
                data = self.provider.fetch_stock_suspend(trade_date=trade_date)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {trade_date} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            logger.info(f"Table {table_name} data update completed.")
        else:
            start_date, end_date = self.provider._get_period(start_date, end_date, update=False)
            tradeday_list = self.calendar.generate(start_date, end_date).strftime("%Y%m%d")
            date_for_query = self.calendar.generate(start_date, end_date).strftime("%Y-%m-%d")
            for trade_date, format_date in tqdm(zip(tradeday_list, date_for_query), total=len(tradeday_list), desc="Normal fetch"):
                if not self.handler._check_data_exists(table_name, format_date):
                    data = self.provider.fetch_stock_suspend(trade_date=trade_date)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {trade_date} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_basic_data(self, table_name, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_basic_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            self.handler._clear_table(table_name)

        data_listed = self.provider.fetch_stock_basic(list_status="L")
        data_delisted = self.provider.fetch_stock_basic(list_status="D")
        data = pd.concat([data_listed, data_delisted], axis=0)
        if not self._check_validity(data):
            logger.warning(f"Stock basic data is empty.")
        self.handler._insert_df_to_table(data, table_name)
        logger.info(f"Table {table_name} data insertion completed.")

    def fetch_stock_income_data(self, table_name="stock_income", ticker_list=None, start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_income_table(table_name)
            self.handler.add_comments(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            # TODO: fix this
            # for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremeantal fetch:"):
            #     start_date = self.handler._get_latest_ticker_date(table_name, ticker)
            #     data = self.provider.fetch_stock_financial_report(ticker=ticker, start_date=start_date, end_date=end_date, fr_type="income")
            #     if not self._check_validity(data):
            #         logger.warning(f"Table {table_name}, {ticker} data is empty.")
            #         continue
            #     self.handler._insert_df_to_table(data, table_name)
            
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_stock_financial_report(ticker=ticker, start_date=start_date, end_date=end_date, fr_type="income")
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            start_date, end_date = self.provider._get_period(start_date, end_date, update=False)
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_financial_report(ticker=ticker, start_date=start_date, end_date=end_date, fr_type="income")
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_balancesheet_data(self, table_name="stock_balancesheet", ticker_list=None, start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_balancesheet_table(table_name)
            self.handler.add_comments(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
        else:
            start_date, end_date = self.provider._get_period(start_date, end_date, update=False)
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_financial_report(ticker=ticker, start_date=start_date, end_date=end_date, fr_type="balancesheet")
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_stock_cashflow_data(self, table_name="stock_cashflow", ticker_list=None, start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_stock_cashflow_table(table_name)
            self.handler.add_comments(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
        else:
            start_date, end_date = self.provider._get_period(start_date, end_date, update=False)
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_stock_financial_report(ticker=ticker, start_date=start_date, end_date=end_date, fr_type="cashflow")
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")

    def fetch_trade_calendar_data(self, table_name, exchange="", start_date=None, end_date=None, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_trade_calendar_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            self.handler._clear_table(table_name)

        data = self.provider.fetch_trade_calendar(exchange=exchange, start_date=start_date, end_date=end_date)
        if not self._check_validity(data):
            logger.warning(f"Trade calendar data is empty.")
        self.handler._insert_df_to_table(data, table_name)
        logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_industry_mapping_data(self, table_name, ticker_list=None, source="SW", update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_industry_mapping_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_industry_mapping(ticker=ticker, source=source)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker, source_str=source):
                    data = self.provider.fetch_industry_mapping(ticker=ticker, source=source)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    continue
            logger.info(f"Table {table_name} data insertion completed.")

    def fetch_index_weight_data(self, table_name="index_weight", ticker_list=None, start_date=None, end_date=None, update=False):
        logger.info(f"Begin to fetch {table_name} data, index count: {len(ticker_list) if ticker_list else 0}, update={update}")
        
        if not self.handler._check_table_exists(table_name):
            self.handler.create_index_weight_table(table_name)
            logger.info(f"Created table {table_name}, storing index weight data.")
        
        if update:
            logger.info(f"Execute incremental update mode for {table_name}")
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremental fetch:"):
                start_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_index_weight(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_index_weight(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            logger.info(f"Execute normal insert mode for {table_name}")
            for ticker in tqdm(ticker_list, desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_index_weight(ticker=ticker, start_date=start_date, end_date=end_date)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    logger.debug(f"Skip {ticker}, data already exists")
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_cb_basic_data(self, table_name, update=False):
        if not self.handler._check_table_exists(table_name):
            self.handler.create_cb_basic_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} data.")
        
        if update:
            self.handler._clear_table(table_name)

        data = self.provider.fetch_convertible_bond_basic()
        if not self._check_validity(data):
            logger.warning(f"Convertible bond basic data is empty.")
        self.handler._insert_df_to_table(data, table_name)
        logger.info(f"Table {table_name} data insertion completed.")
    
    def fetch_cb_daily_data(self, table_name, ticker_list=None, start_date=None, end_date=None, update=False):
        logger.info(f"Begin to fetch {table_name} data, ticker number: {len(ticker_list) if ticker_list else 0}, update={update}")
        # Check if table exists, create table if not exists
        if not self.handler._check_table_exists(table_name):
            self.handler.create_cb_daily_table(table_name)
            logger.info(f"Created table {table_name}, storing {' '.join(table_name.split('_'))} volume-price data.")
        
        # Incremental update based on existing data
        if update:
            logger.info(f"Execute incremental update mode for {table_name}")
            if self.handler._check_table_empty(table_name):
                raise LookupError(f"Table is empty, insert data into table {table_name} first!")
            table_ticker_list = self.handler._get_table_ticker(table_name)
            logger.info(f"Found {len(table_ticker_list)} tickers in the database")
            for ticker in tqdm(table_ticker_list, total=len(table_ticker_list), desc="Incremeantal fetch:"):
                start_date = self.handler._get_latest_ticker_date(table_name, ticker)
                data = self.provider.fetch_convertible_bond_daily(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            # for tickers other than existing ticker
            extra_ticker_list = list(set(ticker_list).difference(table_ticker_list))
            for ticker in tqdm(extra_ticker_list, total=len(extra_ticker_list), desc="Incremeantal extra fetch:"):
                data = self.provider.fetch_convertible_bond_daily(ticker=ticker, start_date=start_date, end_date=end_date, update=True)
                if not self._check_validity(data):
                    logger.warning(f"Table {table_name}, {ticker} data is empty.")
                    continue
                self.handler._insert_df_to_table(data, table_name)
            
            logger.info(f"Table {table_name} data update completed.")
        else:
            logger.info(f"Execute normal insert mode for {table_name}")
            # Perform data extraction, transformation and insertion in a loop
            for ticker in tqdm(ticker_list, total=len(ticker_list), desc="Normal fetch:"):
                if not self.handler._check_data_exists(table_name, ticker):
                    data = self.provider.fetch_convertible_bond_daily(ticker=ticker, start_date=start_date, end_date=end_date)
                    if not self._check_validity(data):
                        logger.warning(f"Table {table_name}, {ticker} data is empty.")
                        continue
                    self.handler._insert_df_to_table(data, table_name)
                else:
                    logger.debug(f"Skip {ticker}, data already exists")
                    continue
            logger.info(f"Table {table_name} data insertion completed.")
    
    def _check_validity(self, data):
        try:
            if data.empty:
                return False
        except:
            if data is None:
                return False
        else:
            return True
    
 