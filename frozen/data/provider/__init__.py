from enum import Enum

class ProviderTypes(Enum):
    TUSHARE = "tushare"
    QMT = "qmt"

class TickerType(Enum):
    INDEX_COMPONENT = "index_component"
    LISTED_STOCK = "listed_stock"
    LISTED_INDEX = "listed_index"
    LISTED_CB = "listed_convertible_bond"