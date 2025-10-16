import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..factor.expression.base import Factor


class StrategyMetric:
    '''Evaluation metrics for the strategy performance.'''

    @staticmethod
    def annual_rate(ret):
        return (1 + ret).prod() ** (252/(len(ret)+1)) - 1

    @staticmethod
    def Beta(rets):
        return rets.cov().iloc[0,1] / rets.cov().iloc[1,1]

    @staticmethod
    def Alpha(rets):
        ar_s = StrategyMetric.annual_rate(rets.Account)
        ar_m = StrategyMetric.annual_rate(rets.Benchmark)
        beta = StrategyMetric.Beta(rets)
        return ar_s - beta * ar_m
    
    @staticmethod
    def Max_Drawdown(net_value):
        end_idx = np.argmax((np.maximum.accumulate(net_value) - net_value) / np.maximum.accumulate(net_value))
        if end_idx == 0:
            return 0
        start_idx = np.argmax(net_value[:end_idx])
        return (net_value[start_idx] - net_value[end_idx]) / net_value[start_idx]
    
    @staticmethod
    def Sharpe_Ratio(ret):
        return ret.mean() / ret.std() * np.sqrt(252)
    
    @staticmethod
    def Sortino_Ratio(ret):
        sigma_d = np.sqrt(1 / len(ret) * (np.where(ret<0, ret, 0) ** 2).sum())
        return ret.mean() / sigma_d * np.sqrt(252)
    
    @staticmethod
    def Information_Ratio(rets):
        diff = rets.Account - rets.Benchmark
        return diff.mean() / diff.std()


class FactorMetric:

    @staticmethod
    def Information_Coefficient(factor: "Factor", inst_close=None, period=1, method="pearson"):
        """
        Calculate the information coefficient between factor value and instrument 1D forward return.
        
        Params:
        -------
        factor: Factor
            Factor class that wraps factor value and expression.
            - Factor value: pd.DataFrame with date as index and instrument ticker as column.
        
        inst_close: pd.DataFrame
            Instrument close price, with date as index and instrument ticker as column.
        
        period: int
            The calculation period of instrument forward returns (default 1D).

        method: str
            The calculation method of correlation coefficient.
            Accepted fields: `pearson` (default), `spearman`, or `kendall`.
        
        Returns:
        --------
        ic: float
            The mean of IC time series (daily).
        """
        
        factor = factor.data
        
        # Calculate forward return based on periods
        inst_return = inst_close.pct_change(period).shift(-period)
        
        # Ensure the index and columns of the two dataframes are the same
        common_dates = factor.index.intersection(inst_return.index)
        common_inst = factor.columns.intersection(inst_return.columns)
        
        factor = factor.loc[common_dates, common_inst]
        inst_return = inst_return.loc[common_dates, common_inst]
        
        # Calculate daily IC
        ic_series = factor.corrwith(inst_return, axis=1, method=method)
        ic = ic_series.mean()
        
        return ic