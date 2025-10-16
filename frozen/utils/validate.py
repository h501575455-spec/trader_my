from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from .helper import unpack_dict_for_init
from .error import FrozenError
from .constants import (
    WeekdayType,
    MonthType,
    WeekdayFreqType,
    MonthFreqType,
    QuarterFreqType,
    YearFreqType,
)
from .constants import (
    SellRuleType,
    SlippageType,
    OptimizerType,
    OptFuncType,
    CovMethodType,
)

class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    """
    
    @abstractmethod
    def validate(self, value) -> bool:
        """
        Validate a given value.
        
        Parameters
        ----------
        value : Any
            The value to validate.
            
        Returns
        -------
        bool
            True if the value is valid, False otherwise.
        """
        pass


class DateRuleValidator(BaseValidator):
    def validate(self, value) -> bool:
        if value is None:
            return False
        try:
            self.parse_rule(value)
            return True
        except FrozenError:
            return False
    
    @staticmethod
    def parse_rule(rule_str: str) -> Union[
        Tuple[int, WeekdayType], 
        Tuple[int, MonthFreqType],
        Tuple[int, QuarterFreqType, MonthType],
        Tuple[int, YearFreqType, MonthType]
    ]:
        """
        Parse the rule string and return the number of frequency,
        frequency type and frequency rule.
        
        The rule string is in the format of "(n)W-MON", "(n)MS",
        "(n)QS-MAR", "(n)YE-FEB" etc.

        Example:
        - "W-MON" -> (1, "W", "MON")
        - "2W-TUE" -> (2, "W", "TUE")
        - "MS" -> (1, "MS", None)
        - "2ME" -> (2, "ME", None)
        - "SMS" -> (1, "SMS", None)
        - "2QS-MAR" -> (2, "QS", "MAR")
        - "3QE-DEC" -> (3, "QE", "DEC")
        - "YS-JAN" -> (1, "YS", "JAN")
        - "2YE-FEB" -> (2, "YE", "FEB")

        Parameters
        ----------
        rule_str: str
            The rule string to be parsed.

        Returns
        -------
        n: int
            The number of frequency.
        type: str
            The type of frequency.
        rule: str
            The rule for frequency type.
        """
        rule_str = rule_str.upper()

        # Handle weekly frequency rule: W-MON, 2W-TUE, etc
        if rule_str.endswith(tuple(day.value for day in WeekdayType)):
            parts = rule_str.split('-')
            period = parts[0]
            weekday = parts[1]
            n = int(period[:-1]) if period[0].isdigit() else 1
            return n, WeekdayFreqType.W, WeekdayType(weekday)
        
        # Handle monthly frequency rule: MS, 2ME, SMS, 3SME, etc
        elif any(rule_str.endswith(freq.value) for freq in MonthFreqType):
            # Match SMS/SME first
            for freq in sorted(MonthFreqType, key=lambda f: -len(f.value)):
                if rule_str.endswith(freq.value):
                    type_str = freq.value
                    digits = rule_str[:-len(type_str)]
                    n = int(digits) if digits.isdigit() else 1
                    return n, MonthFreqType(type_str), None
        
        # Handle quarterly & yearly frequency rule: 2QS-MAR, 3QE-DEC, YS-JAN, 2YE-FEB, etc
        elif rule_str.endswith(tuple(month.value for month in MonthType)):
            parts = rule_str.split('-')
            if len(parts) != 2:
                raise FrozenError(f"Invalid rule format: {rule_str}")
            
            period_part, month_str = parts
            
            # Validate month
            if month_str not in [m.value for m in MonthType]:
                raise FrozenError(f"Invalid month: {month_str}")
            
            # Extract frequency number and type
            n = 1
            freq_type = None
            
            # Check for quarterly frequency type
            if any(ft in period_part for ft in [qt.value for qt in QuarterFreqType]):
                for ft in [qt.value for qt in QuarterFreqType]:
                    if ft in period_part:
                        digits = period_part.replace(ft, '')
                        n = int(digits) if digits.isdigit() else 1
                        freq_type = QuarterFreqType(ft)
                        break
            
            # Check for yearly frequency type
            elif any(ft in period_part for ft in [yt.value for yt in YearFreqType]):
                for ft in [yt.value for yt in YearFreqType]:
                    if ft in period_part:
                        digits = period_part.replace(ft, '')
                        n = int(digits) if digits.isdigit() else 1
                        freq_type = YearFreqType(ft)
                        break
            
            if freq_type:
                return n, freq_type, MonthType(month_str)
            else:
                raise FrozenError(f"Should not pass '{month_str}'")
        
        else:
            raise FrozenError(f"Cannot parse rule: {rule_str}")

    @staticmethod
    def generate_rule(n: int, type_obj: Union[WeekdayType, MonthFreqType, QuarterFreqType, YearFreqType], 
                     month: Optional[MonthType] = None) -> str:
        """
        Generate the date rule string given frequency type.

        Parameters
        ----------
        n: int
            The number of frequency.
        type_obj: Union[WeekdayType, MonthFreqType, QuarterFreqType, YearFreqType]
            The type of frequency.
        month: Optional[MonthType]
            The month for quarterly and yearly frequency.

        Returns
        -------
        rule_str: str
            The date rule string.
        """
        if isinstance(type_obj, WeekdayType):
            # Generate weekly frequency rule: W-MON, 2W-TUE, etc
            return f"{n if n > 1 else ''}W-{type_obj.value}"
        
        elif isinstance(type_obj, MonthFreqType):
            # Generate monthly frequency rule: MS, 2ME, SMS, 3SME, etc
            return f"{n if n > 1 else ''}{type_obj.value}"
        
        elif isinstance(type_obj, (QuarterFreqType, YearFreqType)):
            # Generate quarterly & yearly frequency rule: 2QS-MAR, 3QE-DEC, YS-JAN, 2YE-FEB, etc
            if not month:
                raise FrozenError("Must specify month for quarterly & yearly frequency rule")
            return f"{n if n > 1 else ''}{type_obj.value}-{month.value}"
        
        else:
            raise TypeError("Unsupported type")    


class AssetRangeValidator(BaseValidator):
    def validate(self, value) -> bool:
        if value is None:
            return False
        try:
            return self.validate_tuple(value)
        except FrozenError:
            return False

    @staticmethod
    def validate_tuple(t):
        """
        Validate if a tuple meets the following conditions:
        1. The tuple contains exactly two numbers.
        2. The first number is smaller than the second number.
        3. Both numbers are non-negative.
        4. If a number is a decimal, its value does not exceed 1.
        5. The maximum number does not exceed 200.
        
        Returns:
            True: The tuple meets all conditions.
            False: The tuple does not meet any of the conditions.
        """
        # Check if it is a tuple
        if not isinstance(t, (tuple, list)):
            return False
        
        # Check if the length is 2
        if len(t) != 2:
            return False
        
        # Check if both elements are numbers
        if not (isinstance(t[0], (int, float)) and isinstance(t[1], (int, float))):
            return False
        
        # Check if both numbers are non-negative
        if t[0] < 0 or t[1] < 0:
            return False
        
        # Check if the first number is smaller than the second number
        if t[0] >= t[1]:
            return False
        
        # Check if decimal numbers do not exceed 1
        if isinstance(t[0], float) and t[0] > 1:
            return False
        if isinstance(t[1], float) and t[1] > 1:
            return False
        
        return True


class EnumValidator(BaseValidator):
    def __init__(self, enum_type):
        self.enum_type = enum_type
    
    def validate(self, value) -> bool:
        if value is None:
            if any(item.name == "NULL" for item in self.enum_type):
                return True
            return False
        return value in [item.value for item in self.enum_type]


class WindowValidator(BaseValidator):
    def validate(self, value) -> bool:
        if value is None:
            return True
        try:
            return self.validate_non_negative_int(value)
        except FrozenError:
            return False
    
    def validate_non_negative_int(self, i):
        if not isinstance(i, int):
            return False
        if i < 0:
            return False
        return True


ACCEPTED_FIELDS = {
    "sell_rule": (EnumValidator, SellRuleType),
    "slippage_type": (EnumValidator, SlippageType),
    "optimizer": (EnumValidator, OptimizerType),
    "opt_func": (EnumValidator, OptFuncType),
    "cov_method": (EnumValidator, CovMethodType),
    
    "asset_range": AssetRangeValidator,
    "date_rule": DateRuleValidator,

    "cov_window": WindowValidator,
    "lookback_window": WindowValidator,
}


class ValidatorFactory:
    @staticmethod
    def get_validator(validator_spec):
        if isinstance(validator_spec, tuple):
            validator_cls, arg = validator_spec
            return validator_cls(arg)
        else:
            return validator_spec()


def validate_parameters(params):
    for key, value in params.items():
        if key in ACCEPTED_FIELDS:
            validator_spec = ACCEPTED_FIELDS[key]
            validator = ValidatorFactory.get_validator(validator_spec)
                
            if not validator.validate(value):
                if isinstance(validator, EnumValidator):
                    allowed = [item.value for item in validator.enum_type]
                    raise FrozenError(f"Invalid value for {key}: {value}. Expected one of {allowed}")
                else:
                    raise FrozenError(f"Invalid value for {key}: {value}")         
    return True
