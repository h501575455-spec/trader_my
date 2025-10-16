from frozen.engine import FrozenBt
from frozen.factor import *
from frozen.utils import Universe, Report, AlphaPlot, decorator


class FactorFactory(FrozenBt):


    def __init__(self):

        super().__init__(__file__)
    

    def univ(self):

        universe = Universe(self.config)
        self.universe = universe.pool


    def prepare_data(self):

        data_definitions = [
            ("stock_daily_hfq", ("close", "pct_chg"), ("close", "returns")),
        ]
        
        return self.dataloader.load_batch(data_definitions, self.universe, start_date=self.start_date_lookback, end_date=self.end_date)


    @decorator.factor
    def calc(self):

        # calculate single factor
        string = "cs_rank(ts_argmax(SignedPower(where(returns < 0 ? ts_stddev(returns, 5) : close), 2.0), 5))"
        alpha = calc_str(string, self.prepare_data())

        return alpha


if __name__ == '__main__':

    strategy = FactorFactory()
    strategy.run_backtest(plot=True, plot_type="all", dynamic=False, positions=True, trades=True, html=True)
