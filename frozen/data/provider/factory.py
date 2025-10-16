from . import ProviderTypes
from .tushare.datafeed import TushareDataFeed

class ProviderFactory:
    @staticmethod
    def create_data_feed(
        provider_type: ProviderTypes
    ):
        if provider_type == ProviderTypes.TUSHARE:
            return TushareDataFeed()
        else:
            raise ValueError(f"Unsupported data feed type: {provider_type}")

