import abc
import pandas as pd

class DatabaseHandler(abc.ABC):
    """Abstract base class for database operations"""
    
    @abc.abstractmethod
    def init_db(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def _check_table_exists(self, table_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _check_table_empty(self, table_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _check_data_exists(self, table_name: str, data_str: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_table_ticker(self, table_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _get_table_date(self, table_name: str, latest: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_ticker_date(self, table_date: pd.DataFrame, ticker: str, shift: int):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _insert_df_to_table(self, df: pd.DataFrame, table_name: str):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _delete_table(self, table_name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def _query(self, query_str: str, fmt: str):
        raise NotImplementedError


    def create_volume_price_table(self, table_name):
        pass

    def create_stock_limit_table(self, table_name):
        pass

    def create_stock_fundamental_table(self, table_name):
       pass

    def create_stock_dividend_table(self, table_name):
        pass

    def create_stock_suspend_table(self, table_name):
        pass

    def create_stock_basic_table(self, table_name):
        pass
