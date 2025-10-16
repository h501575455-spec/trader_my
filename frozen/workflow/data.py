from ..data.etl.datafeed import DataFeedManager
from ..data.provider import ProviderTypes, TickerType
from ..data.database import DatabaseTypes
from ..data.provider.factory import ProviderFactory
from ..data.utils.util import parallel_task
from ..utils.calendar import CalendarTypes

class GetData:

    def __init__(
        self,
        provider_type: ProviderTypes,
        database_type: DatabaseTypes,
        calendar_type: CalendarTypes,
    ):
        self.provider = ProviderFactory.create_data_feed(provider_type)
        self.manager = DataFeedManager(
            provider_type=provider_type,
            database_type=database_type,
            calendar_type=calendar_type
        )
        
        # Initialize stock and index tickers
        self.stock_ticker_list = self.provider.get_instrument_list(ticker_type=TickerType.LISTED_STOCK, market="", list_status="L")
        self.index_ticker_list = self.provider.get_instrument_list(ticker_type=TickerType.LISTED_INDEX, market="SSE") + self.provider.get_instrument_list(ticker_type=TickerType.LISTED_INDEX, market="SZSE")
        self.cb_ticker_list = self.provider.get_instrument_list(ticker_type=TickerType.LISTED_CB, exchange="")

    def fetch_basic_info_data(
        self,
        update=False
    ):
        """
        Stock basic data sync
        
        Parameters
        ----------
        update: bool, optional
            Whether to update existing data, defaults to False.
        """
        self.manager.fetch_stock_basic_data("stock_basic", update=update)
        self.manager.fetch_cb_basic_data("convertible_bond_basic", update=update)

    def fetch_daily_price_data(
        self,
        start_date=None,
        end_date=None,
        update=False,
        parallel=False
    ):
        """
        Stock daily price data sync
        
        Parameters
        ----------
        start_date: str, optional
            Start date of data with format YYmmdd, defaults to None.
        end_date: str, optional
            End date of data with format YYmmdd, defaults to None.
        update: bool, optional
            Whether to update existing data, defaults to False.
        parallel: bool
            Whether to use multi-threading during fetch or update processes.
        """
        if not parallel:
            self.manager.fetch_volume_price_data("index_daily", ticker_list=self.index_ticker_list, start_date=start_date, end_date=end_date, asset="I", update=update)
            self.manager.fetch_volume_price_data("stock_daily_real", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)
            self.manager.fetch_volume_price_data("stock_daily_hfq", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, adj="hfq", update=update)
            self.manager.fetch_stock_limit_data("stock_daily_limit", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)
            self.manager.fetch_cb_daily_data("convertible_bond_daily", ticker_list=self.cb_ticker_list, start_date=start_date, end_date=end_date, update=update)
        else:
            stock_tasks = [
                ("index_daily", self.index_ticker_list, start_date, end_date, "I", "", update),
                ("stock_daily_real", self.stock_ticker_list, start_date, end_date, "E", "", update),
                ("stock_daily_hfq", self.stock_ticker_list, start_date, end_date, "E", "hfq", update),
                ("stock_daily_limit", self.stock_ticker_list, start_date, end_date, update),
                ("convertible_bond_daily", self.cb_ticker_list, start_date, end_date, update),
            ]
            parallel_task(self.manager, tasks=stock_tasks)

    def fetch_daily_indicator_data(
        self,
        start_date=None,
        end_date=None,
        update=False,
        parallel=False
    ):
        """
        Stock daily indicator data sync
        
        Parameters
        ----------
        start_date: str, optional
            Start date of data with format YYmmdd, defaults to None.
        end_date: str, optional
            End date of data with format YYmmdd, defaults to None.
        update: bool, optional
            Whether to update existing data, defaults to False.
        parallel: bool
            Whether to use multi-threading during fetch or update processes.
        """
        if not parallel:
            self.manager.fetch_stock_fundamental_data("stock_daily_fundamental", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)
        else:
            stock_tasks = [
                ("stock_daily_fundamental", self.stock_ticker_list, start_date, end_date, update),
            ]
            parallel_task(self.manager, tasks=stock_tasks)

    def fetch_alternative_data(
        self,
        start_date=None,
        end_date=None,
        update=False,
        parallel=False
    ):
        """
        Stock alternative data sync
        
        Parameters
        ----------
        start_date: str, optional
            Start date of data with format YYmmdd, defaults to None.
        end_date: str, optional
            End date of data with format YYmmdd, defaults to None.
        update: bool, optional
            Whether to update existing data, defaults to False.
        parallel: bool
            Whether to use multi-threading during fetch or update processes.
        """
        if not parallel:
            self.manager.fetch_stock_dividend_data("stock_dividend", ticker_list=self.stock_ticker_list, update=update)
            self.manager.fetch_stock_suspend_data("stock_suspend_status", start_date=start_date, end_date=end_date, update=update)
            self.manager.fetch_trade_calendar_data("trade_calendar", exchange="SSE", start_date=start_date, end_date=end_date)
            self.manager.fetch_index_weight_data("index_weight", ticker_list=self.index_ticker_list, start_date=start_date, end_date=end_date, update=update)
        else:
            stock_tasks = [
                ("stock_dividend", self.stock_ticker_list, update),
                ("stock_suspend_status", start_date, end_date, update),
                ("trade_calendar", "SSE", start_date, end_date),
                ("index_weight", self.index_ticker_list, start_date, end_date, update),
            ]
            parallel_task(self.manager, tasks=stock_tasks)

    def fetch_financial_report_data(
        self,
        start_date=None,
        end_date=None,
        update=False,
    ):
        """
        Stock financial data sync
        
        Parameters
        ----------
        start_date: str, optional
            Start date of data with format YYmmdd, defaults to None.
        end_date: str, optional
            End date of data with format YYmmdd, defaults to None.
        update: bool, optional
            Whether to update existing data, defaults to False.
        """
        self.manager.fetch_stock_income_data("stock_income", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)
        self.manager.fetch_stock_balancesheet_data("stock_balancesheet", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)
        self.manager.fetch_stock_cashflow_data("stock_cashflow", ticker_list=self.stock_ticker_list, start_date=start_date, end_date=end_date, update=update)

    def fetch_static_data(
        self,
        update=False
    ):
        """
        Static data sync
        
        Parameters
        ----------
        update: bool, optional
            Whether to update existing data, defaults to False.
        """
        self.manager.fetch_industry_mapping_data("industry_mapping", ticker_list=self.stock_ticker_list, source="SW", update=update)
        self.manager.fetch_industry_mapping_data("industry_mapping", ticker_list=self.stock_ticker_list, source="ZX", update=update)

    def frozen_data(
        self,
        start_date=None,
        end_date=None,
        update=False,
        parallel=False
    ):
        """
        Download frozen data from remote

        Parameters
        ----------
        start_date: str, optional
            Start date of data with format YYmmdd, defaults to None.
        
        end_date: str, optional
            End date of data with format YYmmdd, defaults to None.
        
        update: bool, optional
            Whether to update existing data, defaults to False.
        
        parallel: bool
            Whether to use multi-threading during fetch or update processes.

        Examples
        ---------
        # get daily data
        python -m scripts.get_data --provider TUSHARE --database DUCKDB --calendar CHINA --frozen_data --start_date 20230101 --end_date 20231231
        When this command is run, the data will be downloaded from `tushare`.
        ---------

        """
        self.fetch_basic_info_data(update=update)
        self.fetch_daily_price_data(start_date=start_date, end_date=end_date, update=update, parallel=parallel)
        self.fetch_daily_indicator_data(start_date=start_date, end_date=end_date, update=update, parallel=parallel)
        self.fetch_alternative_data(start_date=start_date, end_date=end_date, update=update, parallel=parallel)
        self.fetch_financial_report_data(start_date=start_date, end_date=end_date, update=update)
        self.fetch_static_data(update=update)
