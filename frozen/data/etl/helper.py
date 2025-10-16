from dataclasses import dataclass
from typing import Union, Tuple, List
from ...utils.calendar import CalendarTypes


@dataclass
class FinancialReportQueryHelper:
    """Base query helper"""
    table_name: str
    ticker: str
    target_date: str
    indicator: Union[str, Tuple[str], List[str]]


@dataclass
class DiscreteQueryHelper(FinancialReportQueryHelper):
    """Discrete data query helper, return a discrete time series of the indicator"""
    date_range_start: str


@dataclass 
class ContinuousQueryHelper(FinancialReportQueryHelper):
    """
    Continuous data query helper, return a continuous time series of the indicator
    
    Parameters
    ----------
    calendar_type: CalendarTypes
        The calendar type to generate continuous financial report data.
    """
    start_date: str
    calendar_type: CalendarTypes


@dataclass
class ValueQueryHelper(FinancialReportQueryHelper):
    """
    Single value query helper, return a value of the indicator
    
    Parameters
    ----------
    period: str
        The report period of the financial report, normally the end of each 
        season, e.g. "20200630" (according to column `end_date` in table)
    """
    period: str

