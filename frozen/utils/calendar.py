import datetime
import holidays
import pandas as pd
import chinese_calendar as cn_calendar

from enum import Enum
from typing import Union, TYPE_CHECKING
from dateutil.relativedelta import relativedelta

from .error import FrozenError
from .validate import DateRuleValidator
from ..data.utils.constants import ONE_DAY, TODAY, DEFAULT_START

if TYPE_CHECKING:
    from ..basis.constant import FrozenConfig


class CalendarTypes(Enum):
    NONE = "null"
    WEEKEND = "weekend"
    CHINA = "cn"
    HONGKONG = "hk"
    UNITED_STATES = "us"
    SSE = "sse"


method_mapping = {
    CalendarTypes.NONE: "generic",
    CalendarTypes.WEEKEND: "generic",
    CalendarTypes.CHINA: "generic",
    CalendarTypes.HONGKONG: "generic",
    CalendarTypes.UNITED_STATES: "generic",
    CalendarTypes.SSE: "query",
}


TRADING_HOURS = {
    CalendarTypes.NONE: [
        {"start": "00:00:00", "end": "23:59:59"}
    ],
    CalendarTypes.CHINA: [
        {"start": "09:30:00", "end": "11:30:00"},  # AM range
        {"start": "13:00:00", "end": "15:00:00"}   # PM range
    ],
    CalendarTypes.UNITED_STATES: [
        {"start": "09:30:00", "end": "16:00:00"}
    ],
    CalendarTypes.HONGKONG: [
        {"start": "09:30:00", "end": "12:00:00"},
        {"start": "13:00:00", "end": "16:00:00"}
    ],
}


class Calendar:
    """The market calendar adjustment module."""

    def __init__(self, cal_type_or_config: Union[CalendarTypes, "FrozenConfig"]):

        if isinstance(cal_type_or_config, CalendarTypes):
            self.cal_type = cal_type_or_config
            self.region = cal_type_or_config.value
            self.config = None

        elif isinstance(cal_type_or_config, self._dynamic_import_frozen_config()):
            self.region = cal_type_or_config.region
            self.cal_type = CalendarTypes(self.region)
            self.config = cal_type_or_config


    @classmethod
    def _dynamic_import_frozen_config(cls):
        from ..basis.constant import FrozenConfig
        return FrozenConfig


    def _parse_time_str(self, time_str, format = "%H:%M:%S"):
        """Parse string to datetime.time object."""
        return datetime.datetime.strptime(time_str, format).time()


    def _get_trading_hours(self):
        trading_hours = TRADING_HOURS.get(self.cal_type, [])
        return trading_hours


    def get_period_delta(self, rule_str):
        """Transform the given frequency rule string to a timedelta 
        that can be directly applied to datetime objects."""

        n, freq, rule = DateRuleValidator.parse_rule(rule_str)

        if freq.value.startswith("W"):
            timedelta = relativedelta(weeks=n)
        elif freq.value.startswith("M"):
            timedelta = relativedelta(months=n)
        elif freq.value.startswith("Q"):
            timedelta = relativedelta(months=3*n)
        elif freq.value.startswith("Y"):
            timedelta = relativedelta(years=n)
        else:
            raise ValueError(f"Invalid frequency: {freq}")
        
        return timedelta


    def is_holiday(self, date):
        """Decide whether the given date is holiday."""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        
        if self.cal_type == CalendarTypes.NONE:
            return False
        
        elif self.cal_type == CalendarTypes.WEEKEND:
            return date.weekday() in holidays.WEEKEND
        
        elif self.cal_type == CalendarTypes.CHINA:
            return cn_calendar.is_holiday(date)
        
        else:
            raise FrozenError(f"Unknown calendar type: {self.cal_type}")


    def is_trade_day(self, date: Union[str, datetime.datetime]):
        """Decide whether the given date is trade day."""
        if isinstance(date, str):
            date = pd.Timestamp(date).date()

        return False if date.weekday() in holidays.WEEKEND or self.is_holiday(date) else True


    def is_trade_time(self, time: Union[str, datetime.datetime]):
        """Decide whether the given time is within trading hours."""
        if isinstance(time, str):
            time = pd.Timestamp(time)
        
        if not self.is_trade_day(time):
            return False
        
        trading_hours = TRADING_HOURS.get(self.cal_type, [])
        time_only = time.time()
        
        for session in trading_hours:
            start_time = self._parse_time_str(session["start"])
            end_time = self._parse_time_str(session["end"])
            
            if start_time <= time_only <= end_time:
                return True
        
        return False
    

    # Deprecated
    # def is_trade_time(self, datetime):
    #     """Check if given datetime is within trading hours."""
    #     if isinstance(datetime, str):
    #         datetime = pd.Timestamp(datetime)
            
    #     # First check if it's a trading day
    #     if not self.is_trade_day(datetime.date()):
    #         return False
            
    #     # Check trading hours (9:30-11:30, 13:00-15:00 for Chinese markets)
    #     time = datetime.time()
    #     if self.region in ["cn", "sse"]:
    #         morning_start = pd.Timestamp("09:30:00").time()
    #         morning_end = pd.Timestamp("11:30:00").time()
    #         afternoon_start = pd.Timestamp("13:00:00").time()
    #         afternoon_end = pd.Timestamp("15:00:00").time()
    #         return (morning_start <= time <= morning_end) or (afternoon_start <= time <= afternoon_end)
    #     elif self.region == "us":
    #         market_start = pd.Timestamp("09:30:00").time()
    #         market_end = pd.Timestamp("16:00:00").time()
    #         return market_start <= time <= market_end
    #     else:
    #         raise ValueError(f"Unsupported region: {self.region}")


    def generate(self, 
                 start_time: Union[str, datetime.datetime] = None, 
                 end_time: Union[str, datetime.datetime] = None, 
                 freq: str = "D"):
        """Generate a list of trade times between the given start time and end time."""

        if start_time is None:
            start_time = DEFAULT_START
        if end_time is None:
            end_time = TODAY

        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)
        
        method = method_mapping.get(self.cal_type, "generic")

        if freq.upper() in ["D", "DAY", "DAILY"]:
            start_date, end_date = start_time.date(), end_time.date()
            if method == "generic":
                # generic calendar method to generate a series of trade days according to given cal_type
                return pd.DatetimeIndex([date for date in pd.date_range(start_date, end_date) if self.is_trade_day(date)])
            elif method == "query":
                # query the database for trade calendars
                from ..data.etl.dataload import DataLoadManager, DatabaseTypes
                dataloader = DataLoadManager(self.config) if self.config else DataLoadManager(DatabaseTypes.DUCKDB)
                return dataloader.load_trade_calendar(exchange=self.region.upper(), start_date=start_date, end_date=end_date)
        else:
            trading_hours = self._get_trading_hours()
            if not trading_hours:
                raise ValueError(f"No trading hours configuration for region {self.region} and frequency {freq}")
            
            start_date = start_time.normalize()
            end_date = end_time.normalize()
            trade_days = self.generate(start_date, end_date, freq="D")

            time_series = []
            for day in trade_days:
                day_str = day.strftime("%Y-%m-%d")
                for session in trading_hours:
                    start_dt = pd.Timestamp(f"{day_str} {session['start']}")
                    end_dt = pd.Timestamp(f"{day_str} {session['end']}")
                    session_times = pd.date_range(
                        start=start_dt, 
                        end=end_dt, 
                        freq=freq
                    )
                    time_series.extend(session_times)
            trade_times = pd.DatetimeIndex(time_series)
            mask = (trade_times >= start_time) & (trade_times <= end_time)
            return trade_times[mask]


    # def generate(self, start_date, end_date):
    #     sse = mcal.get_calendar("SSE")
    #     cal = sse.schedule(start_date=start_date, end_date=end_date, tz="Asia/Shanghai")
    #     return cal.index


    def adjust(self, date, n_days: int = 1):
        """Adjust the given date by the given number of trade days."""
        if isinstance(date, str):
            date = pd.Timestamp(date)

        if n_days == 0:
            return date
        elif n_days > 0:
            for _ in range(n_days):
                date += ONE_DAY
                while not self.is_trade_day(date):
                    date += ONE_DAY
        else:
            for _ in range(-n_days):
                date -= ONE_DAY
                while not self.is_trade_day(date):
                    date -= ONE_DAY
        
        return date
    
    def parse_freq_string(self, freq_str: str) -> str:
        """Parse frequency string like '1m', '5min', '1d' to standard format.
        
        Args:
            freq_str: Frequency string like '1m', '5min', '1h', '1d'
            
        Returns:
            Standard frequency string ('sec', 'min', 'hour', 'day')
        """
        import re
        
        # Remove numbers and get the unit part
        unit = re.sub(r"\d+", "", freq_str.lower())
        
        # Mapping from various formats to standard format
        unit_mapping = {
            "s": "sec",
            "sec": "sec",
            "second": "sec",
            "seconds": "sec",
            "m": "min",
            "min": "min",
            "minute": "min",
            "minutes": "min",
            "h": "hour",
            "hour": "hour",
            "hours": "hour",
            "d": "day",
            "day": "day",
            "days": "day"
        }
        
        if unit in unit_mapping:
            return unit_mapping[unit]
        else:
            raise ValueError(f"Unsupported frequency unit: {unit}")
    
    def adjust_time(self, datetime, n_periods: int = 1, freq: str = "min"):
        """Adjust the given datetime by the given number of periods with specified frequency.
        
        Args:
            datetime: The datetime to adjust
            n_periods: Number of periods to adjust (positive for forward, negative for backward)
            freq: Frequency string ('min', 'hour', 'day', 'sec')
        
        Returns:
            Adjusted datetime
        """
        if isinstance(datetime, str):
            datetime = pd.Timestamp(datetime)
        
        if n_periods == 0:
            return datetime
            
        # Define frequency mappings
        freq_map = {
            "sec": pd.Timedelta(seconds=1),
            "min": pd.Timedelta(minutes=1), 
            "hour": pd.Timedelta(hours=1),
            "day": pd.Timedelta(days=1)
        }
        
        if freq not in freq_map:
            raise ValueError(f"Unsupported frequency: {freq}. Use one of {list(freq_map.keys())}")
        
        delta = freq_map[freq]
        
        if n_periods > 0:
            for _ in range(n_periods):
                datetime += delta
                # Skip non-trading times if frequency is intraday
                if freq in ["sec", "min", "hour"]:
                    while not self.is_trade_time(datetime):
                        datetime += delta
        else:
            for _ in range(-n_periods):
                datetime -= delta
                # Skip non-trading times if frequency is intraday
                if freq in ["sec", "min", "hour"]:
                    while not self.is_trade_time(datetime):
                        datetime -= delta
        
        return datetime
    

    def next_trade_time(self, current_time, freq):
        """Get next trading time right after current time."""
        if not self.is_trade_time(current_time):
            current_date = current_time.date()
            trading_hours = self._get_trading_hours()
            
            for session in trading_hours:
                session_start = datetime.datetime.combine(
                    current_date, self._parse_time_str(session["start"])
                )
                session_end = datetime.datetime.combine(
                    current_date, self._parse_time_str(session["end"])
                )
                
                if current_time < session_start:
                    return session_start
                
                if current_time < session_end:
                    time_points = pd.date_range(
                        start=current_time, 
                        end=session_end, 
                        freq=freq
                    )
                    if len(time_points) > 0:
                        next_point = time_points[0]
                        if next_point > current_time:
                            return next_point
                        elif len(time_points) > 1:
                            return time_points[1]
            
            next_day = current_date + datetime.timedelta(days=1)
            while not self.is_trade_day(next_day):
                next_day += datetime.timedelta(days=1)
            
            first_session_start = datetime.datetime.combine(
                next_day, self._parse_time_str(trading_hours[0]["start"])
            )
            return first_session_start
        
        current_date = current_time.date()
        trading_hours = self._get_trading_hours()
        
        for i, session in enumerate(trading_hours):
            session_start = datetime.datetime.combine(
                current_date, self._parse_time_str(session["start"])
            )
            session_end = datetime.datetime.combine(
                current_date, self._parse_time_str(session["end"])
            )
            
            if session_start <= current_time <= session_end:
                current_session_idx = i
                break
        
        if current_session_idx >= 0:
            session = trading_hours[current_session_idx]
            session_start = datetime.datetime.combine(
                current_date, self._parse_time_str(session["start"])
            )
            session_end = datetime.datetime.combine(
                current_date, self._parse_time_str(session["end"])
            )
            
            time_points = pd.date_range(
                start=current_time, 
                end=session_end, 
                freq=freq
            )
            
            if len(time_points) > 1:
                return time_points[1]
            
            if current_session_idx < len(trading_hours) - 1:
                next_session = trading_hours[current_session_idx + 1]
                next_session_start = datetime.datetime.combine(
                    current_date, self._parse_time_str(next_session["start"])
                )
                return next_session_start
        
        next_day = current_date + datetime.timedelta(days=1)
        while not self.is_trade_day(next_day):
            next_day += datetime.timedelta(days=1)
        
        first_session_start = datetime.datetime.combine(
            next_day, self._parse_time_str(trading_hours[0]["start"])
        )
        return first_session_start


    def next_trade_dates(self, dates):
        """Generate a list of following trade dates."""
        
        return pd.DatetimeIndex([self.adjust(date, n_days=1) for date in dates])


    def previous_trade_dates(self, dates):
        """Generate a list of previous trade dates."""

        return pd.DatetimeIndex([self.adjust(date, n_days=-1) for date in dates])


    def exist_holiday_between_dates(self, start_date, end_date):
        """Determine if there's holiday between the given dates."""
        next_day = start_date + ONE_DAY
        while next_day <= end_date:
            if next_day.weekday() not in holidays.WEEKEND and self.is_holiday(next_day):
                return True
            else:
                next_day += ONE_DAY
        return False

