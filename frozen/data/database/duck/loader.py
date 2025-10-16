import pandas as pd
from typing import Tuple, List, Union
from ..factory import DatabaseFactory, DatabaseTypes
from ....utils.calendar import CalendarTypes
from ...etl.helper import (
    DiscreteQueryHelper, 
    ContinuousQueryHelper, 
    ValueQueryHelper
)
from .pit import (
    PITDataGenerator, 
    BatchPITQueryEngine, 
    VectorizedPITQueryEngine, 
    HyperPITQueryEngine, 
    UltraPITQueryEngine
)

class DuckDBLoader:
    
    def __init__(self):
        self.handler = DatabaseFactory.create_database_connection(DatabaseTypes.DUCKDB)
        self.handler.init_db()

    def load_time_series_data(self, table_name, column, universe, start_date, end_date):
        if column is not None and not isinstance(column, (str, tuple, list)):
            raise TypeError("Unsupported type for 'column'! Only string, tuple and list are allowed.")
        
        if column is None:
            col_str = "*"
        else:
            if isinstance(column, (tuple, list)):
                unique_columns = list(dict.fromkeys(column))
                col_str = ", ".join(unique_columns)
            else:
                col_str = column
            col_str = "ticker, trade_date, " + col_str
        
        if universe is None:
            univ_str = ""
        else:
            if isinstance(universe, str):
                univ_str = f"ticker == '{universe}' AND" 
            elif isinstance(universe, (tuple, list)):
                univ_str = f"ticker IN {universe} AND"
            else:
                raise TypeError("Unsupported type for 'universe'! Only string, tuple and list are allowed.")
        
        start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        query_str = f"""
            SELECT {col_str} FROM {table_name} 
            WHERE {univ_str} trade_date >= '{start_date}' AND trade_date <= '{end_date}' 
            ORDER BY trade_date DESC
            """
        data = self.handler._query(query_str, read_only=True)
        return data
    
    def load_time_series_data_high_freq(self, table_name, column, universe, start_time, end_time):
        if column is not None and not isinstance(column, (str, tuple, list)):
            raise TypeError("Unsupported type for 'column'! Only string, tuple and list are allowed.")
        
        if column is None:
            col_str = "*"
        else:
            if isinstance(column, (tuple, list)):
                unique_columns = list(dict.fromkeys(column))
                col_str = ", ".join(unique_columns)
            else:
                col_str = column
            col_str = "ticker, datetime, " + col_str
        
        if universe is None:
            univ_str = ""
        else:
            if isinstance(universe, str):
                univ_str = f"ticker == '{universe}' AND" 
            elif isinstance(universe, (tuple, list)):
                univ_str = f"ticker IN {universe} AND"
            else:
                raise TypeError("Unsupported type for 'universe'! Only string, tuple and list are allowed.")
        
        start_time = pd.to_datetime(start_time).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")
        
        query_str = f"""
            SELECT {col_str} FROM {table_name} 
            WHERE {univ_str} datetime >= '{start_time}' AND datetime <= '{end_time}' 
            ORDER BY datetime DESC
            """
        data = self.handler._query(query_str, read_only=True)
        return data
    
    def load_basic_data(self, table_name):
        query_str = f"SELECT * FROM {table_name}"
        data = self.handler._query(query_str, read_only=True)
        return data
    
    def load_dividend_data(self, table_name, universe):
        query_str = f"SELECT * FROM {table_name} WHERE ticker IN {universe}"
        data = self.handler._query(query_str, read_only=True)
        return data
    
    def load_suspend_data(self, table_name, start_date, end_date):
        start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        query_str = f"SELECT * FROM {table_name} WHERE trade_date>='{start_date}' AND trade_date<='{end_date}'"
        data = self.handler._query(query_str, read_only=True)
        return data
    
    def load_financial_report_data_multiple(self, table_name: str, universe: List[str], target_date: str, start_date: str, indicator: Union[str, Tuple[str], List[str]], calendar_type: CalendarTypes = CalendarTypes.NONE):
        helpers = [
            ContinuousQueryHelper(
                table_name=table_name,
                ticker=ticker,
                target_date=target_date,
                indicator=indicator,
                start_date=start_date,
                calendar_type=calendar_type
            )
            for ticker in universe
        ]
        data = pd.concat({helper.ticker: self.load_financial_report_data_single(helper) for helper in helpers}, axis=0, names=["ticker", "trade_date"])
        return data
    
    def load_financial_report_data_single(
            self,
            helper: Union[DiscreteQueryHelper, ContinuousQueryHelper, ValueQueryHelper]
        ):
        pit_engine = PITDataGenerator(
            helper.table_name, 
            helper.ticker,
            helper.target_date,
            helper.indicator
        )
        
        if isinstance(helper, ContinuousQueryHelper):
            data = pit_engine.get_historical_continuous_series(
                helper.start_date,
                helper.calendar_type
            )
        elif isinstance(helper, DiscreteQueryHelper):
            data = pit_engine.get_historical_discrete_series(
                helper.date_range_start
            )
        elif isinstance(helper, ValueQueryHelper):
            data = pit_engine.get_historical_value(
                helper.period, 
            )
        else:
            raise ValueError(f"Invalid helper type: {type(helper)}")
        return data
    
    def load_financial_report_data_slice(self, table_name: str, universe: List[str], target_dates: List[str], indicators: Union[str, List[str]], lookback_months: int = 12, expanding: bool = True, discrete: bool = True, method: str = "ultra", **kwargs):
        if method == "ultra":
            pit_engine = UltraPITQueryEngine(table_name)
        elif method == "hyper":
            pit_engine = HyperPITQueryEngine(table_name)
        elif method == "vectorized":
            pit_engine = VectorizedPITQueryEngine(table_name)
        elif method == "batch":
            pit_engine = BatchPITQueryEngine(table_name)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if discrete:
            data = pit_engine.get_discrete_series(
                tickers=universe,
                target_dates=target_dates,
                indicators=indicators,
                lookback_months=lookback_months,
                expanding=expanding
            )
        else:
            calendar_type = kwargs.get(calendar_type, CalendarTypes.NONE)
            data = pit_engine.get_continuous_series(
                tickers=universe,
                target_dates=target_dates,
                indicators=indicators,
                lookback_months=lookback_months,
                expanding=expanding,
                calendar_type=calendar_type
            )
        return data

    def load_trade_calendar(self, exchange="SSE", start_date=None, end_date=None):
        start_date = pd.Timestamp(start_date) if start_date else pd.Timestamp.min
        end_date = pd.Timestamp(end_date) if end_date else pd.Timestamp.max
        query_str = """
            SELECT cal_date
            FROM trade_calendar
            WHERE is_open == 1
                AND exchange = ?
                AND cal_date BETWEEN ? AND ?
            """
        data = self.handler._query(
            query_str,
            params=(exchange, start_date, end_date),
            read_only=True
        )
        return data
    
    def load_index_weight(self, index_code: str, target_date: Union[str, pd.Timestamp]):
        target_date = pd.Timestamp(target_date)
        query_str = """
            SELECT con_code, weight FROM index_weight
            WHERE ticker = ?
            AND trade_date = (
                SELECT MAX(trade_date) 
                FROM index_weight 
                WHERE ticker = ? 
                AND trade_date <= ?
            )
            """
        data = self.handler._query(
            query_str,
            params=(index_code, index_code, target_date),
            read_only=True
        )
        return data
    
    def load_column(self, table_name):
        query_str = f"""
            SELECT *
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            """
        data = self.handler._query(query_str)
        return data
    
    def load_industry_mapping(self, classification: str):
        """
        classification: str
        - 'basic': unknown source
        - 'sw_l1': 申万一级行业
        - 'sw_l2': 申万二级行业
        - 'sw_l3': 申万三级行业
        - 'zx_l1': 中信一级行业
        - 'zx_l2': 中信二级行业
        - 'zx_l3': 中信三级行业
        """
        if classification == "basic":
            query_str = "SELECT ticker, industry FROM stock_list"
        else:
            source, level = tuple(classification.split("_"))
            source, level = source.upper(), level.lower()
            query_str = f"""
                SELECT ticker, {level}_name
                FROM industry_mapping
                WHERE source='{source}'
                """
        data = self.handler._query(query_str, read_only=True)
        return data

