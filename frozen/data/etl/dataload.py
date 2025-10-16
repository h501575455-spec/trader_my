import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Tuple, List

from ...basis import FrozenConfig
from ..database import DatabaseTypes
from ..database.factory import DatabaseFactory
from .helper import (
    DiscreteQueryHelper, 
    ContinuousQueryHelper, 
    ValueQueryHelper
)
from ...factor.expression import Factor
from ...utils.calendar import Calendar, CalendarTypes
from ..utils.constants import (
    DEFAULT_START, DEFAULT_END, DEFAULT_START_TIME, DEFAULT_END_TIME
)

class DataLoader(ABC):
    """
    DataLoader is designed for loading raw data from original data source.
    """

    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]):
        """
        Initialize DataLoader with either DatabaseTypes or FrozenConfig
        
        Parameters:
        -----------
        database_type_or_config: Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        if isinstance(database_type_or_config, DatabaseTypes):
            # If DatabaseTypes is provided, use it and create default config
            self.database_type = database_type_or_config
            self.config = FrozenConfig()  # Create default config instance
            self.calendar = Calendar(CalendarTypes.NONE)
        elif isinstance(database_type_or_config, FrozenConfig):
            # If FrozenConfig is provided, extract database type from it
            self.config = database_type_or_config
            self.database_type = DatabaseTypes(self.config.database)
            self.calendar = Calendar(self.config)
        else:
            raise TypeError(f"Expected DatabaseTypes or FrozenConfig, got {type(database_type_or_config)}")
        
        self.loader = DatabaseFactory.create_data_loader(self.database_type)

    @abstractmethod
    def load(
        self,
    ) -> pd.DataFrame:
        
        raise NotImplementedError
    
    
    def _check_validity(self, data):

        is_empty = False
        
        if data is None:
            is_empty = True
        elif isinstance(data, pd.DataFrame):
            is_empty = data.empty
        elif isinstance(data, list):
            is_empty = len(data) == 0
        elif isinstance(data, dict):
            is_empty = len(data) == 0
        
        if is_empty:
            raise ValueError(f"Missing data detected. Please check the integrity of {self.database_type.value} database entries.")


class TSDataLoader(DataLoader):
    """
    (T)ime-(S)eries DataLoader.
    """

    def __init__(
        self,
        database_type_or_config: Union[DatabaseTypes, FrozenConfig],
    ) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(
        self,
        table_name: str,
        column: Union[str, Tuple, List] = None,
        universe: Union[str, Tuple, List] = None,
        start_date: str = None,
        end_date: str = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        table_name: str
            Load data from what table in the source database.
        column: Union[str, Tuple, List]
            The columns to be selected.
            If not specified, all columns will be loaded.
        universe: Union[str, Tuple, List]
            It marks the pool of instrument tickers to be loaded.
            If not specified, all universe will be loaded.
        start_date: str
            The start date of the time range, in format `YYYYmmdd`.
        end_date: str
            The end date of the time range, in format `YYYYmmdd`.
        **kwargs: dict
            -- lookback: int
                The lookback period from end date.
            -- multiindex: bool
                To return a multi-index dataframe or not.
            -- fillna: bool
                To fill NA values or not.
        Returns
        -------
        pd.DataFrame:
            The data loaded from the under layer source.
        """

        lookback = kwargs.get("lookback", None)
        multiindex = kwargs.get("multiindex", False)
        fillna = kwargs.get("fillna", False)

        if lookback is not None:
            start_date = self.calendar.adjust(end_date, -lookback)
        
        # Handle column=None case for multiindex
        processed_col = column
        if multiindex and column is None:
            temp_data = self._data_transformer(table_name, column, universe, start_date, end_date)
            processed_col = tuple(temp_data.columns)
        
        data = self._data_handler(table_name, column, universe, start_date, end_date, fillna=fillna)

        if multiindex:
            data = self._restore_to_multiindex(data, processed_col)

        return data
    
    def _data_loader(
        self,
        table_name: str,
        column: Union[str, Tuple, List],
        universe: Union[str, Tuple, List],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:

        if start_date is None and end_date is None:
            start_date, end_date = DEFAULT_START, DEFAULT_END
        elif not start_date is None and end_date is None:
            start_date, end_date = start_date, DEFAULT_END
        elif start_date is None and not end_date is None:
            start_date, end_date = DEFAULT_START, end_date
        else:
            start_date, end_date = start_date, end_date
        
        data = self.loader.load_time_series_data(table_name, column, universe, start_date, end_date)
        self._check_validity(data)

        return data


    def _data_transformer(
        self,
        table_name: str,
        column: Union[str, Tuple, List],
        universe: Union[str, Tuple, List],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Transform data into multi-index format
        - level_0: ticker
        - level_1: trade_date
        """

        data = self._data_loader(table_name, column, universe, start_date, end_date)
        # drop duplicated rows
        data.drop_duplicates(keep="first", inplace=True)
        # data transformation
        data.set_index(["ticker", "trade_date"], inplace=True)
        data.sort_index(level=0, ascending=True, inplace=True)

        return data
    

    def _data_handler(
        self,
        table_name: str,
        column: Union[str, Tuple, List],
        universe: Union[str, Tuple, List],
        start_date: str,
        end_date: str,
        fillna: bool = False
    ) -> pd.DataFrame:
        """
        Pipeline that load stock daily data (both volume-price 
        and fundamental) from database and transform into formatted 
        dataframe.
        """

        data = self._data_transformer(table_name, column, universe, start_date, end_date)

        def _process_column(data):
            result = data.swaplevel().unstack()
            return result.ffill() if fillna else result
        
        if column is None:
            column = tuple(data.columns)
        
        if isinstance(column, str):
            return _process_column(data[column])
        else:  # tuple and list
            return tuple(_process_column(data[c]) for c in column)

    def _restore_to_multiindex(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame]], column: Union[str, Tuple, List]) -> pd.DataFrame:
        """
        Convert unstacked DataFrame(s) back to multiindex format.
        
        Parameters
        ----------
        data: Union[pd.DataFrame, Tuple[pd.DataFrame]]
            The unstacked data from _data_handler
        column: Union[str, Tuple, List]
            The column specification used in _data_handler
            
        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with (ticker, trade_date) index
        """
        if isinstance(column, str):
            # Single column case
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for single column, got {type(data)}")
            return data.stack().swaplevel().sort_index().to_frame(column)
        
        else:  # tuple case
            if not isinstance(data, tuple):
                raise TypeError(f"Expected tuple of DataFrames for multiple columns, got {type(data)}")
            
            # Convert each DataFrame back to multiindex Series and combine
            multiindex_series = []
            for df, column_name in zip(data, column):
                series = df.stack().swaplevel().sort_index()
                series.name = column_name
                multiindex_series.append(series)
            
            # Combine all series into a single DataFrame
            return pd.concat(multiindex_series, axis=1) if column else None


class BasicDataLoader(DataLoader):
    """
    Stock Basic DataLoader

    DataLoader that supports loading stock basic data.
    """

    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(self, table_name: str) -> pd.DataFrame:
        
        data = self._basic_loader(table_name)
        
        return data
    
    def _basic_loader(self, table_name) -> pd.DataFrame:

        data = self.loader.load_basic_data(table_name)
        self._check_validity(data)

        return data


class DividendDataLoader(DataLoader):
    """
    Stock Dividend DataLoader

    DataLoader that supports loading stock dividend data.
    """

    def __init__(
        self,
        database_type_or_config: Union[DatabaseTypes, FrozenConfig],
    ) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(
            self,
            table_name: str,
            universe: Tuple = (),
            **kwargs
        ) -> pd.DataFrame:
        
        data = self._dividend_loader(table_name, universe, **kwargs)

        return data
    

    def _dividend_loader(self, table_name, universe: Tuple = (), **kwargs) -> dict:

        if not universe:
            raise ValueError("arg `universe` must not be none.")

        format = kwargs.get("format", "dict")

        raw_data = self.loader.load_dividend_data(table_name, universe)
        self._check_validity(raw_data)
        data = self._transform_dividend_data(raw_data)

        if format == "dict":
            return data

        elif format == "dataframe":
            collapsed_data = (
                pd.concat(data, keys=data.keys(), names=["ticker"])
                .reset_index(level=1)
                .set_index("ex_date", append=True)
                .drop(columns="level_1")
            )
            return collapsed_data
    
    def _transform_dividend_data(self, raw_data):
        data = {
            key: (
                group
                # Step 1: screen out the data with `div_proc` as `实施`
                .query("div_proc=='实施'")
                .drop(["ticker"], axis=1)
                .sort_values(by="ex_date")
                .reset_index(drop=True)
                # Step 2: sum the data by `ex_date`
                .groupby("ex_date", as_index=False)
                .agg({
                    **{col: "sum" for col in ["stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"]},  # take the sum
                    # **{col: "first" for col in ["stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"]}  # keep first value
                })
            )
            for key, group in raw_data.groupby("ticker")
        }
        return data


class SuspendDataLoader(DataLoader):
    """
    Stock Suspend DataLoader

    DataLoader that supports loading stock suspend data.
    """

    def __init__(
        self,
        database_type_or_config: Union[DatabaseTypes, FrozenConfig],
    ) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(
            self,
            table_name: str,
            start_date: str = None,
            end_date: str = None
        ) -> pd.DataFrame:
        
        data = self._suspend_loader(table_name, start_date, end_date)

        return data
    
    
    def _suspend_loader(self, table_name, start_date, end_date) -> pd.DataFrame:

        start_date = DEFAULT_START if start_date is None else start_date
        end_date = DEFAULT_END if end_date is None else end_date

        raw_data = self.loader.load_suspend_data(table_name, start_date, end_date)
        self._check_validity(raw_data)
        data = self._transform_suspend_data(raw_data)
    
        return data
    
    def _transform_suspend_data(self, raw_data):
        data = raw_data.sort_values(by=["trade_date", "ticker"], ascending=True)
        data.set_index("trade_date", inplace=True)
        return data


class FinancialReportDataLoader(DataLoader):
    """
    Financial Report DataLoader

    DataLoader that supports loading financial report data.
    """

    def __init__(
        self,
        database_type_or_config: Union[DatabaseTypes, FrozenConfig],
    ) -> None:
        super().__init__(database_type_or_config)
    
    def load(self):
        pass

    def load_single_ticker(
        self,
        helper: Union[DiscreteQueryHelper, ContinuousQueryHelper, ValueQueryHelper]
    ) -> pd.DataFrame:
        """
        Load financial report data for single ticker using helper object.
        
        Parameters
        ----------
        helper: Union[DiscreteQueryHelper, ContinuousQueryHelper, ValueQueryHelper]
            Query helper object, return different types of data based on different helper types:
            - DiscreteQueryHelper: return discrete time series data
            - ContinuousQueryHelper: return continuous time series data (need calendar_type)
            - ValueQueryHelper: return single value of the indicator (need period)
        """
        data = self.loader.load_financial_report_data_single(helper)
        self._check_validity(data)
        return data
    
    def load_multiple_tickers(
        self,
        table_name: str,
        universe: Tuple = (),
        target_date: str = None,
        start_date: str = None,
        indicator: Union[str, Tuple[str], List[str]] = None,
        calendar_type: CalendarTypes = CalendarTypes.NONE
    ) -> pd.DataFrame:
        """
        Load financial report data for multiple tickers using helper object.
        """
        data = self.loader.load_financial_report_data_multiple(table_name, universe, target_date, start_date, indicator, calendar_type)
        self._check_validity(data)
        return data
    
    def load_multiple_tickers_slice(
        self,
        table_name: str,
        universe: Tuple = (),
        target_dates: List[str] = None,
        indicators: Union[str, List[str]] = None,
        lookback_months: int = 12,
        expanding: bool = True,
        discrete: bool = True,
        method: str = "ultra",
        **kwargs
    ) -> pd.DataFrame:
        """
        Load financial report data for multiple tickers with point-in-time (PIT) slicing.
        """
        data = self.loader.load_financial_report_data_slice(table_name, universe, target_dates, indicators, lookback_months, expanding, discrete, method, **kwargs)
        self._check_validity(data)
        return data


class TradeCalendarLoader(DataLoader):
    """Trade Calendar DataLoader"""

    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(self, exchange="SSE", start_date=None, end_date=None) -> pd.DataFrame:
        
        data = self._calendar_loader(exchange, start_date, end_date)
        
        return data
    
    def _calendar_loader(self, exchange, start_date, end_date) -> pd.DataFrame:

        raw_calendar = self.loader.load_trade_calendar(exchange, start_date, end_date)
        self._check_validity(raw_calendar)
        calendar = self._transform_trade_calendar(raw_calendar)

        return calendar
    
    def _transform_trade_calendar(self, raw_calendar):
        calendar_series = raw_calendar["cal_date"].sort_values()
        return pd.DatetimeIndex(calendar_series)


class ColumnLoader(DataLoader):
    
    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]) -> None:
        super().__init__(database_type_or_config)
    
    def load(self, table_name: str, info: str):
        column_info = self.loader.load_column(table_name)
        if info == "name":
            column_info = column_info[["column_name"]]
        if info == "comment":
            column_info = column_info[["column_name", "COLUMN_COMMENT"]]
        return column_info


class IndustryMappingLoader(DataLoader):
    
    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]) -> None:
        super().__init__(database_type_or_config)
    
    def load(self, classification: str):
        data = self._industry_loader(classification)
        return data
    
    def _industry_loader(self, classification):
        raw_data = self.loader.load_industry_mapping(classification)
        self._check_validity(raw_data)
        data = self._transform_industry_data(raw_data)
        return data
    
    def _transform_industry_data(self, raw_data):
        raw_data.columns = ["ticker", "industry"]
        return raw_data


class IndexWeightLoader(DataLoader):

    def __init__(self, database_type_or_config: Union[DatabaseTypes, FrozenConfig]) -> None:
        super().__init__(database_type_or_config)
    
    def load(self, index_code, target_date, code_only: bool = False):
        data = self._index_weight_loader(index_code, target_date, code_only)
        return data
    
    def _index_weight_loader(self, index_code, target_date, code_only):
        data = self.loader.load_index_weight(index_code, target_date)
        self._check_validity(data)
        if code_only:
            data = self._transform_index_weight_data(data)
        return data
    
    def _transform_index_weight_data(self, raw_data):
        data = raw_data.con_code.tolist()
        return data


class HighFreqDataLoader(DataLoader):
    """
    High Frequency DataLoader.
    """

    def __init__(
        self,
        database_type_or_config: Union[DatabaseTypes, FrozenConfig],
    ) -> None:
        """
        Parameters
        ----------
        database_type_or_config : Union[DatabaseTypes, FrozenConfig]
            Either a DatabaseTypes enum or a FrozenConfig instance
        """
        super().__init__(database_type_or_config)

    def load(
        self,
        table_name: str,
        column: Union[str, Tuple, List] = None,
        universe: Union[str, Tuple, List] = None,
        start_time: str = None,
        end_time: str = None,
        freq: str = "1m",
        **kwargs
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        table_name: str
            Load data from what table in the source database.
        column: Union[str, Tuple, List]
            The columns to be selected.
            If not specified, all columns will be loaded.
        universe: Union[str, Tuple, List]
            It marks the pool of instrument tickers to be loaded.
            If not specified, all universe will be loaded.
        start_time: str
            The start time of the time range, in format `YYYYmmdd HH:MM:SS`.
        end_time: str
            The end time of the time range, in format `YYYYmmdd HH:MM:SS`.
        freq: str
            The frequency of the data, in format `1m` or `1d`.
        **kwargs: dict
            -- lookback: int
                The lookback period from end time.
            -- multiindex: bool
                To return a multi-index dataframe or not.
            -- fillna: bool
                To fill NA values or not.
        Returns
        -------
        pd.DataFrame:
            The data loaded from the under layer source.
        """

        lookback = kwargs.get("lookback", None)
        multiindex = kwargs.get("multiindex", False)
        fillna = kwargs.get("fillna", False)

        if lookback is not None:
            start_time = self.calendar.adjust_time(end_time, -lookback, freq=self.calendar.parse_freq_string(freq))
        
        # Handle column=None case for multiindex
        processed_col = column
        if multiindex and column is None:
            temp_data = self._data_transformer_high_freq(table_name, column, universe, start_time, end_time)
            processed_col = tuple(temp_data.columns)
        
        data = self._data_handler_high_freq(table_name, column, universe, start_time, end_time, fillna=fillna)

        if multiindex:
            data = self._restore_to_multiindex(data, processed_col)

        return data
    
    def _data_loader_high_freq(
        self,
        table_name: str,
        column: Union[str, Tuple, List] = None,
        universe: Union[str, Tuple, List] = None,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:

        if start_time is None and end_time is None:
            start_time, end_time = DEFAULT_START_TIME, DEFAULT_END_TIME
        elif not start_time is None and end_time is None:
            start_time, end_time = start_time, DEFAULT_END_TIME
        elif start_time is None and not end_time is None:
            start_time, end_time = DEFAULT_START_TIME, end_time
        else:
            start_time, end_time = start_time, end_time
        
        data = self.loader.load_time_series_data_high_freq(table_name, column, universe, start_time, end_time)
        self._check_validity(data)

        return data


    def _data_transformer_high_freq(
        self,
        table_name: str,
        column: Union[str, Tuple, List] = None,
        universe: Union[str, Tuple, List] = None,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:
        """
        Transform data into multi-index format
        - level_0: ticker
        - level_1: datetime
        """

        data = self._data_loader_high_freq(table_name, column, universe, start_time, end_time)
        # drop duplicated rows
        data.drop_duplicates(keep="first", inplace=True)
        # data transformation
        data.set_index(["ticker", "datetime"], inplace=True)
        data.sort_index(level=0, ascending=True, inplace=True)

        return data
    

    def _data_handler_high_freq(
        self,
        table_name: str,
        column: Union[str, Tuple, List] = None,
        universe: Union[str, Tuple, List] = None,
        start_time: str = None,
        end_time: str = None,
        fillna: bool = False
    ) -> pd.DataFrame:
        """
        Pipeline that load stock daily data (both volume-price 
        and fundamental) from database and transform into formatted 
        dataframe.
        """

        data = self._data_transformer_high_freq(table_name, column, universe, start_time, end_time)
        
        def _process_column(data):
            result = data.swaplevel().unstack()
            return result.ffill() if fillna else result
        
        if column is None:
            column = tuple(data.columns)
        
        if isinstance(column, str):
            return _process_column(data[column])
        else:  # tuple and list
            return tuple(_process_column(data[c]) for c in column)

    def _restore_to_multiindex(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame]], column: Union[str, Tuple, List]) -> pd.DataFrame:
        """
        Convert unstacked DataFrame(s) back to multiindex format.
        
        Parameters
        ----------
        data: Union[pd.DataFrame, Tuple[pd.DataFrame]]
            The unstacked data from _data_handler
        column: Union[str, Tuple, List]
            The column specification used in _data_handler
            
        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with (ticker, datetime) index
        """
        if isinstance(column, str):
            # Single column case
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for single column, got {type(data)}")
            return data.stack().swaplevel().sort_index().to_frame(column)
        
        else:  # tuple case
            if not isinstance(data, tuple):
                raise TypeError(f"Expected tuple of DataFrames for multiple columns, got {type(data)}")
            
            # Convert each DataFrame back to multiindex Series and combine
            multiindex_series = []
            for df, column_name in zip(data, column):
                series = df.stack().swaplevel().sort_index()
                series.name = column_name
                multiindex_series.append(series)
            
            # Combine all series into a single DataFrame
            return pd.concat(multiindex_series, axis=1) if column else None


class DataLoadManager:

    def __init__(self, database_type_or_config: Union[str, DatabaseTypes, FrozenConfig]):
        """
        Initialize DataLoadManager with either database string, DatabaseTypes or FrozenConfig
        
        Parameters:
        -----------
        database_type_or_config: Union[str, DatabaseTypes, FrozenConfig]
            Either a database type string, DatabaseTypes enum or a FrozenConfig instance
        """
        if isinstance(database_type_or_config, str):
            self.database_type = DatabaseTypes(database_type_or_config)
            self.config = FrozenConfig()  # Create default config instance
        elif isinstance(database_type_or_config, DatabaseTypes):
            self.database_type = database_type_or_config
            self.config = FrozenConfig()  # Create default config instance
        elif isinstance(database_type_or_config, FrozenConfig):
            self.config = database_type_or_config
            self.database_type = DatabaseTypes(self.config.database)
        else:
            raise TypeError(f"Expected str or FrozenConfig, got {type(database_type_or_config)}")
    
    def load_volume_price(
            self,
            table_name: str,
            column: Union[str, Tuple, List] = None,
            universe: Union[str, Tuple, List] = None,
            start_date: str = None,
            end_date: str = None,
            **kwargs
        ):

        loader = TSDataLoader(self.database_type)
        data = loader.load(table_name, column, universe, start_date, end_date, **kwargs)

        return data
    
    def load_bar(self, table_name: str, column: Union[str, Tuple] = None, universe: Tuple = (), start_time: str = None, end_time: str = None, freq: str = "1m", **kwargs):
        if freq == "1m":
            loader = HighFreqDataLoader(self.config)
            data = loader.load(table_name, column, universe, start_time, end_time, freq, **kwargs)
        elif freq == "1d":
            loader = TSDataLoader(self.database_type)
            data = loader.load(table_name, column, universe, start_time, end_time, **kwargs)
        else:
            raise ValueError(f"Frequency {freq} not supported")
        return data
    
    def load_basic_info(self, table_name: str):

        loader = BasicDataLoader(self.database_type)
        data = loader.load(table_name)

        return data
    
    def load_stock_dividend(self, table_name, universe: Tuple = (), **kwargs):
        
        loader = DividendDataLoader(self.database_type)
        data = loader.load(table_name, universe, **kwargs)

        return data
    
    def load_stock_suspend(self, table_name, start_date, end_date):

        loader = SuspendDataLoader(self.database_type)
        data = loader.load(table_name, start_date, end_date)

        return data
    
    def load_financial_report_single(
            self, 
            helper: Union[DiscreteQueryHelper, ContinuousQueryHelper, ValueQueryHelper]
        ):
        
        loader = FinancialReportDataLoader(self.database_type)
        data = loader.load_single_ticker(helper)
        return data
    
    def load_financial_report_multiple(
            self,
            table_name: str,
            universe: Tuple = (),
            target_date: str = None,
            start_date: str = None,
            indicator: Union[str, Tuple[str], List[str]] = None,
            calendar_type: CalendarTypes = CalendarTypes.NONE
        ):
        loader = FinancialReportDataLoader(self.database_type)
        data = loader.load_multiple_tickers(table_name, universe, target_date, start_date, indicator, calendar_type)
        return data
    
    def load_financial_report_pit_slice(
        self,
        table_name: str,
        universe: Tuple = (),
        target_dates: List[str] = None,
        indicators: Union[str, List[str]] = None,
        lookback_months: int = 12,
        expanding: bool = True,
        discrete: bool = True,
        method: str = "ultra",
        **kwargs
    ):
        loader = FinancialReportDataLoader(self.database_type)
        data = loader.load_multiple_tickers_slice(table_name, universe, target_dates, indicators, lookback_months, expanding, discrete, method, **kwargs)
        return data

    def load_trade_calendar(self, exchange: str = "SSE", start_date: str = None, end_date: str = None):
        loader = TradeCalendarLoader(self.database_type)
        data = loader.load(exchange, start_date, end_date)
        return data

    def load_column_schema(self, table_name: str, info: str = "name"):
        loader = ColumnLoader(self.database_type)
        column = loader.load(table_name, info)
        return column
    
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
        loader = IndustryMappingLoader(self.database_type)
        data = loader.load(classification)
        return data
    
    def loade_index_weight(self, index_code: str, target_date: str, code_only: bool = False):
        loader = IndexWeightLoader(self.database_type)
        data = loader.load(index_code, target_date, code_only)
        return data

    def _ts_wrap(self, table_name, columns, names, universe=None, start_date=None, end_date=None, **kwargs):
        """
        Load time-series data and wrap into Factor.
        """

        data_list = self.load_volume_price(table_name, columns, universe, start_date, end_date, **kwargs)

        if isinstance(data_list, tuple):
            return [Factor(data, name) for data, name in zip(data_list, names)]
        else:
            return Factor(data_list, names)

    def _dividend_wrap(self, table_name, indicator, name, universe=None, start_date=None, end_date=None, **kwargs):
        """
        Load dividend data and wrap into Factor.
        """
        raw_data = self.load_stock_dividend(table_name, universe, format="dataframe", **kwargs)
        
        if raw_data.empty:
            return Factor(pd.DataFrame(), name)
        
        # Transform to unstacked format like volume-price data
        data = raw_data[indicator].swaplevel().unstack()
        return Factor(data, name)
    
    def _industry_wrap(self, classification, name, universe=None, **kwargs):
        """
        Load industry data and wrap into Factor.
        """
        from ...utils.industry import get_industry_manager
        
        industry_manager = get_industry_manager(database_type=self.database_type)
        industry_df = industry_manager.get_stock_industry_mapping(classification=classification)
        industry_df = industry_df.drop_duplicates(subset=["ticker"], keep="first")
        
        if universe is not None:
            # Filter to universe if specified
            industry_df = industry_df[industry_df["ticker"].isin(universe)]
            industry_df = industry_df.reset_index(drop=True)
        
        return Factor(industry_df, name)

    def _financial_report_wrap(self, table_name, indicators, names, universe=None, **kwargs):
        """
        Load financial report data and wrap into Factor(s).
        
        Supports tables: stock_income, stock_cashflow, stock_balancesheet
        """
        target_dates = kwargs.get("target_dates")
        lookback_months = kwargs.get("lookback_months")
        expanding = kwargs.get("expanding")

        if target_dates is None:
            raise ValueError("target_dates is required for financial report data")
        
        # Validate table name
        valid_tables = ["stock_income", "stock_cashflow", "stock_balancesheet"]
        if table_name not in valid_tables:
            raise ValueError(f"table_name must be one of {valid_tables}, got {table_name}")
        
        raw_data = self.load_financial_report_pit_slice(table_name, universe, target_dates, indicators, lookback_months, expanding, discrete=True)
        
        if raw_data.empty:
            if isinstance(indicators, str):
                return Factor(pd.DataFrame(), names)
            else:
                return [Factor(pd.DataFrame(), name) for name in names]
        
        # Keep the three-dimensional multiindex as-is
        # Index: (ticker, target_date, end_date) -> indicators
        if isinstance(indicators, str):
            # Single indicator - return as Factor with multiindex structure
            return Factor(raw_data[indicators], names)
        else:
            # Multiple indicators - return list of Factors with multiindex structure
            factors = []
            for indicator, name in zip(indicators, names):
                factors.append(Factor(raw_data[indicator], name))
            return factors

    def load_batch(self, data_def: list, default_universe=None, start_date=None, end_date=None, batch_config=None, **kwargs):
        """
        (Batch) Load data as Factor
        
        Supports multiple data types with automatic detection:
        1. Time-series data (volume-price, fundamental): (table, cols, names, *universe)
        2. Dividend data: ("stock_dividend", indicator, name, *universe)
        3. Financial report data: (fr_table, indicators, names, *universe)
           where fr_table is one of: "stock_income", "stock_cashflow", "stock_balancesheet"
        4. Industry data: ("industry_mapping", classification, name, *universe)
           where classification is one of: "basic", "sw_l1", "sw_l2", "sw_l3", "zx_l1", "zx_l2", "zx_l3"
        
        Parameters:
        -----------
        data_def: list
            List of data definitions. Format is automatically detected based on table name and parameters:
            - Time-series: (table, cols, names, *universe)
            - Dividend: ("stock_dividend", indicator, name, *universe)  
            - Financial report: (fr_table, indicators, names, *universe)
            - Industry: ("industry_mapping", classification, name, *universe)
        default_universe: optional
            Default universe to use when not specified in data_def
        start_date, end_date: str
            Date range for time-series and dividend data
        config: dict, optional
            Configuration dictionary for financial report data. Should include:
            - target_dates: Target date series for PIT data retrieve
            - lookback_months: Determine start date for the first observation
            - expanding: Whether to use expanding window
            - calendar_type: CalendarType to generate continuous dates
        **kwargs: additional parameters
        
        Returns:
        --------
        dict: {name: Factor} mapping
        
        Examples:
        ---------
        data_def = [
            # Time-series data
            ("stock_price", ("close", "volume"), ("close", "volume")),
            
            # Dividend data  
            ("stock_dividend", "stk_div", "dividend"),
            
            # Income statement data
            ("stock_income", ("n_income", "revenue"), ("net_income", "revenue")),
            
            # Industry data
            ("industry_mapping", "sw_l1", "industry"),
        ]
        """
        def wrapped_ts_wrap(table, columns, names, universe=None, **kwargs):
            if universe is None:
                universe = default_universe
            return self._ts_wrap(table, columns, names, universe, start_date, end_date, **kwargs)
        
        def wrapped_dividend_wrap(table, indicator, name, universe=None, **kwargs):
            if universe is None:
                universe = default_universe
            return self._dividend_wrap(table, indicator, name, universe, start_date, end_date, **kwargs)
        
        def wrapped_financial_report_wrap(table, indicators, names, universe=None, **kwargs):
            if universe is None:
                universe = default_universe
            return self._financial_report_wrap(table, indicators, names, universe, **kwargs)
        
        def wrapped_industry_wrap(classification, name, universe=None, **kwargs):
            if universe is None:
                universe = default_universe
            return self._industry_wrap(classification, name, universe, **kwargs)
        
        vars = {}
        
        # Financial report table names
        financial_report_tables = ["stock_income", "stock_cashflow", "stock_balancesheet"]

        # Retrieve parameters for financial report from config
        if batch_config is None:
            batch_config = {}
        
        target_dates = batch_config.get("target_dates")
        lookback_months = batch_config.get("lookback_months")
        expanding = batch_config.get("expanding")
        
        fr_kwargs = {
            "target_dates": target_dates,
            "lookback_months": lookback_months,
            "expanding": expanding,
        }
        
        for data_spec in data_def:
            if len(data_spec) < 3:
                raise ValueError(f"Invalid data_spec format: {data_spec}")
                
            table_name = data_spec[0]
            
            if table_name == "stock_dividend":
                # Dividend data: ("stock_dividend", indicator, name, *universe)
                if len(data_spec) < 3:
                    raise ValueError("Dividend data_spec must have format: ('stock_dividend', indicator, name, *universe)")
                
                _, indicator, name, *universe_spec = data_spec
                factor = wrapped_dividend_wrap(table_name, indicator, name, *universe_spec, **kwargs)
                vars[name] = factor
                
            elif table_name == "industry_mapping":
                # Industry data: ("industry_mapping", classification, name, *universe)
                if len(data_spec) < 3:
                    raise ValueError("Industry data_spec must have format: ('industry_mapping', classification, name, *universe)")
                
                _, classification, name, *universe_spec = data_spec
                factor = wrapped_industry_wrap(classification, name, *universe_spec, **kwargs)
                vars[name] = factor
                
            elif table_name in financial_report_tables:
                # Financial report data: (fr_table, indicators, names, *universe)
                if len(data_spec) < 3:
                    raise ValueError(f"Financial report data_spec must have format: ('{table_name}', indicators, names, *universe)")
                
                _, indicators, names, *universe_spec = data_spec
                
                if isinstance(indicators, str):
                    # Single indicator
                    factor = wrapped_financial_report_wrap(table_name, indicators, names, *universe_spec, **fr_kwargs)
                    vars[names] = factor
                else:
                    # Multiple indicators
                    factors = wrapped_financial_report_wrap(table_name, indicators, names, *universe_spec, **fr_kwargs)
                    for name, factor in zip(names, factors):
                        vars[name] = factor
                        
            else:
                # Time-series data (original logic): (table, cols, names, *universe)
                table, columns, names, *universe_spec = data_spec
                if isinstance(columns, tuple):
                    factors = wrapped_ts_wrap(table, columns, names, *universe_spec, **kwargs)
                    for (name, factor) in zip(names, factors):
                        vars[name] = factor
                else:
                    vars[names] = wrapped_ts_wrap(table, columns, names, *universe_spec, **kwargs)

        return vars