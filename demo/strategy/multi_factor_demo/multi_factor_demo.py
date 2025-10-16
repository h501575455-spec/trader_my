from frozen.engine import FrozenBt
from frozen.factor import *
from frozen.utils import Universe


class FactorFactory(FrozenBt):

    def __init__(self):
        # Configuration
        super().__init__(__file__)

    def univ(self):
        """Define instrument universe on start"""
        universe = Universe(self.config)
        self.universe = universe.pool

    def prepare_data(self):
        """Prepare related data to be used in factor calculation"""
        data_definitions = [
            ('stock_daily_hfq', ('close', 'pct_chg'), ('close', 'returns')),
            ]
        return self.dataloader.load_batch(data_definitions, self.universe, start_date=self.start_date_lookback, end_date=self.end_date)

    def calc(self):
        """Calculate and wrap factor data"""
        str_list = []
        name_list = []

        str_list += ['cs_rank(ts_argmax(SignedPower(where(returns < 0 ? ts_stddev(returns, 5) : close), 2.0), 5))']
        name_list += ['alpha1']

        str_list += ['ts_argmax(SignedPower(where(returns < 0 ? ts_stddev(returns, 5) : close), 2.0), 5)']
        name_list += ['alpha2']

        str_list += ['normalize(ts_argmax(SignedPower(where(returns < 0 ? ts_stddev(returns, 5) : close), 2.0), 5), cross_section=True)']
        name_list += ['alpha3']

        alpha = batch_calc(str_list, name_list, self.prepare_data())
        return alpha

if __name__ == "__main__":
    factory = FactorFactory()
    alpha = factory.calc()
    print(alpha)