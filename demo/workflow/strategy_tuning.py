from frozen.engine import FrozenBt
from frozen.factor import *
from frozen.utils import Universe
from frozen.utils.autotune import StrategyTuning


class FactorFactory(FrozenBt, StrategyTuning):

    def __init__(self):
        super().__init__(__file__)
        StrategyTuning.__init__(self)
    
    def univ(self):
        universe = Universe(self.config)
        self.universe = universe.pool

    def prepare_data(self):
        data_definitions = [
            ("stock_daily_hfq", ("open", "high", "low", "close", "change", "amount", "volume"), ("open", "high", "low", "close", "change", "amount", "vol")),
        ]
        return self.dataloader.load_batch(data_definitions, self.universe, self.start_date_lookback, self.end_date)

    def calc(self):
        alpha_str = "where(ts_min(delta(close, 1), 5) > 0 ? delta(close, 1) : where((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (mul(delta(close, 1), -1))))"
        alpha = calc_str(alpha_str, self.prepare_data())
        return alpha


if __name__ == "__main__":

    test = FactorFactory()
    PARAM_SPACE = {
    "asset_range": {
        "type": "tuple",
        "lower_min": 0,
        "lower_max": 5,
        "upper_min": 10,
        "upper_max": 20
    },
    "date_rule": {
        "type": "categorical",
        "choice": ["W-WED", "ME", "2W-MON", "QE-MAR"]
    },
    "optimizer": {
        "type": "categorical",
        "choice": ["equal-weight", "mean-variance"]
    },
    "opt_func": {
        "type": "categorical",
        "choice": ["sharpe", "variance"],
        "dependent": "optimizer",
        "conditions": {
            "equal-weight": None,
            "mean-variance": ["sharpe", "variance"]
        }
    },
    "cov_method": {
        "type": "categorical",
        "choice": ["custom", "shrink", "poet", "structured"],
        "dependent": "optimizer",
        "conditions": {
            "equal-weight": None,
            "mean-variance": ["custom", "shrink", "poet"]
        }
    },
    "cov_window": {
        "type": "int",
        "dependent": "optimizer",
        "conditions": {
            "equal-weight": None,
            "mean-variance": {
                "min": 30,
                "max": 120,
                "step": 1,
                "log": False
            }
        }
    }
}
    test.autotune(PARAM_SPACE, n_trials=5, n_jobs=-1)

