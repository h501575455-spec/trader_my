from abc import ABC, abstractmethod

class DataFeed(ABC):

    @abstractmethod
    def fetch_volume_price(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_stock_limit(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_stock_fundamental(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_stock_dividend(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_stock_basic(self, *args, **kwargs):
        pass

    @abstractmethod
    def fetch_stock_suspend(self, *args, **kwargs):
        pass
