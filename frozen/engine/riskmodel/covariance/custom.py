import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Literal, Union

from . import RiskModel

MAT_TYPE = Literal["historical", "single-index", "port-est"]


class CustomCovEstimator(RiskModel):

    MT_H: MAT_TYPE = "historical"
    MT_S: MAT_TYPE = "single-index"
    MT_P: MAT_TYPE = "port-est"

    def __init__(self, mat_type: Union[str, int] = 'historical', X_mkt: Union[np.ndarray, pd.Series] = None,  **kwargs):
        """
        Args:
            mat_type (str | int): covariance matrix type to use, which could be:
                - 'historical': sample covariance matrix using historical returns.
                - 'single-index': single-index matrix by one-factor model.
                - 'port-est': portfolio of estimators.
            X_mkt (np.ndarray | pd.Series): time-series of market return
            kwargs: see `RiskModel` for more information.
        
        Note:
            - `X_mkt` is only applicable when `mat_type` is `single-index` or `port-est`

        References:
            [1] 
        """
        super().__init__(**kwargs)

        if isinstance(mat_type, str):
            assert mat_type in [
                self.MT_H,
                self.MT_S,
                self.MT_P,
            ], f"covariance matrix type `{mat_type} is not supported"
        elif isinstance(mat_type, int):
            pass
        else:
            raise TypeError("invalid argument type for `mat_type`")
        self.mat_type = mat_type
        
        if mat_type in [self.MT_S, self.MT_P]:
            assert X_mkt is not None, "argument `X_mkt` cannot be `None` if `mat_type` is `single-index` or `port-est`"
            if isinstance(X_mkt, (pd.Series, pd.DataFrame)):
                X_mkt = X_mkt.values
            elif isinstance(X_mkt, np.ndarray):
                pass
            else:
                raise TypeError("invalid argument type for `X_mkt`")
        self.X_mkt = X_mkt
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate covariance matrix estimation
        """
        if self.mat_type == self.MT_H:
            cov = np.cov(X, rowvar=False, bias=False)
        
        elif self.mat_type == self.MT_S:
            cov = self._gen_single_index(X, self.X_mkt)
        
        elif self.mat_type == self.MT_P:
            sample_cov = np.cov(X, rowvar=False, bias=False)
            diag_cov = np.diag(np.var(X, axis=0, ddof=1))
            single_cov = self._gen_single_index(X, self.X_mkt)
            cov = 1/3 * sample_cov + 1/3 * diag_cov + 1/3 * single_cov
        
        return cov
    
    def _gen_single_index(self, stock, market):
        """
        Generate single index estimator
        """
        y, X = stock, market
        X = sm.add_constant(X)
        est = sm.OLS(y,X).fit()
        beta = est.params[-1].reshape(-1,1)
        resid_var = np.var(y - est.predict(X), axis=0, ddof=1)
        
        mkt_var = np.var(market)
        sgl_idx = mkt_var * np.dot(beta, beta.T) + np.diag(resid_var)

        return sgl_idx
    