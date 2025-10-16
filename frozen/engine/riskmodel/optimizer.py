# Portfolio Optimization module

import abc
import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Literal, Union
from functools import partial
from ...basis import FrozenConfig
from ..riskmodel.covariance import *
from ...utils import unpack_dict_for_init


class BaseOptimizer(metaclass=abc.ABCMeta):
    '''Optimization based on target functions'''

    @abc.abstractmethod
    def target_func(self, *args, **kwargs) -> object:
        '''Define target functions to be optimized'''
    
    @abc.abstractmethod
    def calc_portfolio_weights(self, *args, **kwargs) -> object:
        '''Output optimal weights from optimization algorithms'''
    
    def __call__(self, *args, **kwargs) -> object:
        """leverage Python syntactic sugar to make the models' behaviors like functions"""
        return self.calc_portfolio_weights(*args, **kwargs)

OPT_TYPE = Literal["equal-weight", "mean-variance"]
TGT_FUNC = Literal["sharpe", "variance"]

class PortOpt(BaseOptimizer):

    OPT_EW: OPT_TYPE = "equal-weight"
    OPT_MV: OPT_TYPE = "mean-variance"

    TGT_S: TGT_FUNC = "sharpe"
    TGT_V: TGT_FUNC = "variance"

    def __init__(self, config: FrozenConfig):
        """Initialize optimization parameters"""
        unpacked_dict = unpack_dict_for_init(config.all_config.get("portfolio_config"))
        self.__dict__.update(unpacked_dict)

    def __repr__(self):
        """This method returns a string representation of a Port_Opt object"""
        return "Portfolio_Optimization"
    
    def calc_portfolio_weights(self, n, X, **kwargs):
        """
        This method returns the result of portfolio optimization, i.e., portfolio weights.
        """
        eweights = np.array(n * [1./n])

        if self.optimizer == self.OPT_EW:
            return eweights
        
        elif self.optimizer == self.OPT_MV:
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            if not self.long_short:
                bnds = tuple((0, 1) for _ in range(n))
            else:
                bnds = tuple((-1, 1) for _ in range(n))
            
            partial_target_func = partial(self.target_func, **kwargs)
            opts = sco.minimize(partial_target_func, eweights, method='SLSQP', args=(X,), bounds=bnds, constraints=cons)

            w = opts['x']

        return w
    
    def _covariance_estimator(self, X: Union[np.ndarray, pd.DataFrame], **kwargs):
        """
        Configure covariance matrix estimator.
        ------
        Args:
            X (np.ndarray | pd.DataFrame): time-series of instrument returns.
            kwargs: extra arguments needed to calculate covariacne matrix, e.g., X_mkt.
        """
        if self.cov_method == 'custom':
            if 'X_mkt' in kwargs.keys():
                cov = CustomCovEstimator(mat_type='port-est', X_mkt=kwargs.get('X_mkt'))
            else:
                cov = CustomCovEstimator(mat_type='historical')
        elif self.cov_method == 'shrink':
            cov = ShrinkCovEstimator(alpha='lw', target='const_var')
        elif self.cov_method == 'poet':
            cov = POETCovEstimator()
        elif self.cov_method == 'structured':
            cov = StructuredCovEstimator()
        elif self.cov_method == 'barra':
            # 添加Barra风险模型支持
            barra_params = kwargs.get('barra_params', {})
            cov = BarraRiskModel(**barra_params)
        return cov.predict(X)
    
    def target_func(self, weights, X, **kwargs):
        """Define target functions"""
        if self.opt_func == self.TGT_S:
            func = self._max_sharpe(weights, X, **kwargs)
        elif self.opt_func == self.TGT_V:
            func = self._min_var(weights, X, **kwargs)
        else:
            raise NotImplementedError('Optimization function not implemented yet!')
        return func
    
    def _max_sharpe(self, weights: np.ndarray, X: pd.DataFrame, **kwargs):
        """Maximize Sharpe"""
        return - np.sum(X.mean() * weights) * 252 + 20 * np.sqrt(np.dot(weights.T, np.dot(self._covariance_estimator(X, **kwargs) * 252, weights)))
    
    def _min_var(self, weights: np.ndarray, X: pd.DataFrame, **kwargs):
        """Minimize variance"""
        return np.sqrt(np.dot(weights.T, np.dot(self._covariance_estimator(X, **kwargs) * 252, weights)))
    
    def get_barra_risk_attribution(self, weights: pd.Series, X: pd.DataFrame, **kwargs) -> dict:
        """
        使用Barra风险模型进行风险归因分析
        
        Parameters:
        -----------
        weights : pd.Series
            组合权重
        X : pd.DataFrame
            收益数据
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        dict
            风险归因结果
        """
        if self.cov_method != 'barra':
            raise ValueError("风险归因分析需要使用Barra风险模型")
        
        barra_params = kwargs.get('barra_params', {})
        barra_model = BarraRiskModel(**barra_params)
        
        # 预测协方差矩阵（这会计算所有必要的组件）
        barra_model.predict(X)
        
        # 进行风险归因分析
        risk_attribution = barra_model.risk_attribution(weights)
        
        return risk_attribution
    
    def barra_stress_test(self, weights: pd.Series, X: pd.DataFrame, stress_scenarios: dict, **kwargs) -> dict:
        """
        使用Barra风险模型进行压力测试
        
        Parameters:
        -----------
        weights : pd.Series
            组合权重
        X : pd.DataFrame
            收益数据
        stress_scenarios : dict
            压力测试情景
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        dict
            压力测试结果
        """
        if self.cov_method != 'barra':
            raise ValueError("压力测试需要使用Barra风险模型")
        
        barra_params = kwargs.get('barra_params', {})
        barra_model = BarraRiskModel(**barra_params)
        
        # 预测协方差矩阵（这会计算所有必要的组件）
        barra_model.predict(X)
        
        # 进行压力测试
        stress_results = barra_model.stress_test(weights, stress_scenarios)
        
        return stress_results
