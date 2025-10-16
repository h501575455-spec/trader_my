from frozen.engine import FrozenBt
from frozen.factor import *
from frozen.utils import Universe, AlphaPlot
from frozen.utils.autotune import FactorTuning


class FactorFactory(FrozenBt, FactorTuning):

    def __init__(self):
        super().__init__(__file__)
        FactorTuning.__init__(self)

    def univ(self):
        universe = Universe(self.config)
        self.universe = universe.pool

    def prepare_data(self):
        data_definitions = [
            ("stock_daily_hfq", ("open", "high", "low", "close", "change", "amount", "volume"), ("open", "high", "low", "close", "change", "amount", "vol")),
        ]
        return self.dataloader.load_batch(data_definitions, self.universe, self.start_date_lookback, self.end_date)

    def create_alpha_str(self, params):
        alpha_str = f"""where(ts_min(delta(close, 1), {params['window1']}) > 0 ? delta(close, 1) : where((ts_max(delta(close, 1), {params['window2']}) < 0) ? delta(close, 1) : (mul(delta(close, 1), {params['multiplier']}))))"""
        return alpha_str

    def calc(self, params):
        alpha_str = self.create_alpha_str(params)
        alpha = calc_str(alpha_str, self.prepare_data())
        return alpha

    def evaluate_best_alpha(self, best_alpha_str):
        alpha = calc_str(best_alpha_str, self.prepare_data())
        al_plot = AlphaPlot(self.config, alpha, self.inst_close)
        al_plot.factor_layers(long_short=True, n_groups=5)

    def run_best_alpha(self, best_alpha_str):
        self.alpha = calc_str(best_alpha_str, self.prepare_data())
        self._calc_flag = True
        self.run_backtest()


if __name__ == '__main__':

    test = FactorFactory()
    PARAM_SPACE = {
        'window1': {
            'type': 'int',
            'min': 2,
            'max': 10
        },
        'window2': {
            'type': 'int',
            'min': 2,
            'max': 10
        },
        'multiplier': {
            'type': 'float',
            'min': -2.0,
            'max': -0.1
        }
    }
    raw_params = {}
    _, _, _, best_alpha_str = test.autotune(param_space=PARAM_SPACE, params=raw_params, n_trials=10, n_jobs=-1)
    test.evaluate_best_alpha(best_alpha_str)
    test.run_best_alpha(best_alpha_str)
