from frozen.engine import FrozenBt
from frozen.factor import Factor
from frozen.utils import FileReader

class FactorFactory(FrozenBt):
    """
    A customizable file-based factor strategy.
    """

    def __init__(self):
        self.factor_data = FileReader.read("test_pred.csv", index_col="trade_date", parse_dates=["trade_date"])
        super().__init__(__file__, config_set={"start_date": self.factor_data.index.min().strftime("%Y%m%d"), "end_date": self.factor_data.index.max().strftime("%Y%m%d")})

    def univ(self):
        self.universe = tuple(self.factor_data.columns.unique())

    def prepare_data(self):
        pass

    def calc(self):
        return Factor(self.factor_data)


if __name__ == "__main__":
    factory = FactorFactory()
    factory.run_backtest(plot=True, plot_type="line", positions=True, trades=True, html=True)