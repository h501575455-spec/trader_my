from enum import Enum

class WeekdayType(Enum):
    MON = "MON"
    TUE = "TUE"
    WED = "WED"
    THU = "THU"
    FRI = "FRI"

class MonthType(Enum):
    JAN = "JAN"
    FEB = "FEB"
    MAR = "MAR"
    APR = "APR"
    MAY = "MAY"
    JUN = "JUN"
    JUL = "JUL"
    AUG = "AUG"
    SEP = "SEP"
    OCT = "OCT"
    NOV = "NOV"
    DEC = "DEC"

class WeekdayFreqType(Enum):
    W = "W"

class MonthFreqType(Enum):
    MS = "MS"  # Month Start
    ME = "ME"  # Month End
    SMS = "SMS"  # Semi-Month Start
    SME = "SME"  # Semi-Month End

class QuarterFreqType(Enum):
    QS = "QS"  # Quarter Start
    QE = "QE"  # Quarter End

class YearFreqType(Enum):
    YS = "YS"  # Year Start
    YE = "YE"  # Year End

class SellRuleType(Enum):
    CURRENT_BAR = "current-bar"
    NEXT_BAR = "next-bar"

class SlippageType(Enum):
    PERCENTAGE = "percentage"
    ABSOLUTE = "absolute"

class TransactionType(Enum):
    BUY = "buy"
    SELL = "sell"

class OptimizerType(Enum):
    EQUAL_WEIGHT = "equal-weight"
    MEAN_VARIANCE = "mean-variance"

class OptFuncType(Enum):
    SHARPE = "sharpe"
    VARIANCE = "variance"
    NULL = "null"

class CovMethodType(Enum):
    CUSTOM = "custom"
    SHRINK = "shrink"
    POET = "poet"
    STRUCTURED = "structured"
    NULL = "null"
