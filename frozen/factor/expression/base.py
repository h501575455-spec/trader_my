# Factor Expression Engine

from ...data.database import DatabaseTypes
from ...utils.calendar import CalendarTypes

class Factor:
    """
    Factor base class, which includes two attributes: 
    Factor.data and Factor.expr, defined respectively 
    to achieve the functionality of factor expression 
    engine.

    Perform base factor operations by reloading python 
    magic methods inside Factor class. The module supports 
    recoding the following magic methods:
    
    - Unary Operators
        - Basic operator
            __pos__: x
            __neg__: -x
            __abs__: abs(x)
            __invert__: ~x
        - Type conversion
            __int__: int(x)
            __float__: float(x)
            __index__: bin(x), oct(x), hex(x)
        - Math functions
            __round__: round(x)
            __trunc__: math.trunc(x)
            __floor__: math.floor(x)
            __ceil__: math.ceil(x)
    
    - Binary Operators:
        - Arithmetic operator
            __add__: x + y
            __iadd__: x += y
            __radd__: y + x
            __sub__: x - y
            __rsub__: y - x
            __mul__: x * y
            __rmul__: y * x
            __matmul__: x @ y
            __truediv__: x / y
            __rtruediv__: y / x
            __floordiv__: x // y
            __mod__: x % y
            __divmod__: divmod(x, y)
            __pow__: x ** y
            __lshift__: x << y
            __rshift__: x >> y
            __complex__: complex(x, y)
        - Logical operator
            __and__: x & y
            __or__: x | y
            __xor__: x ^ y

    """
    
    def __init__(self, data=None, expr=""):
        
        self.data = data
        self.expr = expr


    def __str__(self):

        return self.expr
    

    def __pos__(self):

        return Factor(+self.data, f"+{self.expr}")
    

    def __neg__(self):

        return Factor(-self.data, "- " + self.expr)
    

    def __abs__(self):

        return Factor(abs(self.data), f"abs({self.expr})")
    

    def __eq__(self, other):
        
        if isinstance(other, (int, float)):
            eq_data = self.data == other
            eq_expr = f"({self.expr} = {other})"

        elif isinstance(other, Factor):
            eq_data = self.data == other.data
            eq_expr = f"({self.expr} = {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for ==: 'Factor' and '{type(other)}'")
        
        return Factor(eq_data, eq_expr)
    

    def __gt__(self, other):
        
        if isinstance(other, (int, float)):
            gt_data = self.data > other
            gt_expr = f"({self.expr} > {other})"

        elif isinstance(other, Factor):
            gt_data = self.data > other.data
            gt_expr = f"({self.expr} > {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for >: 'Factor' and '{type(other)}'")
        
        return Factor(gt_data, gt_expr)
    

    def __lt__(self, other):
        
        if isinstance(other, (int, float)):
            lt_data = self.data < other
            lt_expr = f"({self.expr} < {other})"

        elif isinstance(other, Factor):
            lt_data = self.data < other.data
            lt_expr = f"({self.expr} < {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for <: 'Factor' and '{type(other)}'")
        
        return Factor(lt_data, lt_expr)


    def __add__(self, other):

        if isinstance(other, (int, float)):
            add_data = self.data + other
            add_expr = f"({self.expr} + {other})"

        elif isinstance(other, Factor):
            add_data = self.data + other.data
            add_expr = f"({self.expr} + {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Factor' and '{type(other)}'")
        
        return Factor(add_data, add_expr)
    

    def __radd__(self, other):

        if isinstance(other, (int, float)):
            add_data = other + self.data
            add_expr = f"({other} + {self.expr})"
        
        elif isinstance(other, Factor):
            add_data = other.data + self.data
            add_expr = f"({other.expr} + {self.expr})"
        
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(other)}' and 'Factor'")
        
        return Factor(add_data, add_expr)
    
    
    def __sub__(self, other):

        if isinstance(other, (int, float)):
            sub_data = self.data - other
            sub_expr = f"({self.expr} - {other})"

        elif isinstance(other, Factor):
            sub_data = self.data - other.data
            sub_expr = f"({self.expr} - {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for -: 'Factor' and '{type(other)}'")
        
        return Factor(sub_data, sub_expr)
    

    def __rsub__(self, other):

        if isinstance(other, (int, float)):
            sub_data = other - self.data
            sub_expr = f"({other} - {self.expr})"
        
        elif isinstance(other, Factor):
            sub_data = other.data - self.data
            sub_expr = f"({other.expr} - {self.expr})"
        
        else:
            raise TypeError(f"Unsupported operand type(s) for -: '{type(other)}' and 'Factor'")
        
        return Factor(sub_data, sub_expr)
    
    
    def __mul__(self, other):

        if isinstance(other, (int, float)):
            mul_data = self.data * other
            mul_expr = f"({self.expr} * {other})"

        elif isinstance(other, Factor):
            mul_data = self.data * other.data
            mul_expr = f"({self.expr} * {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Factor' and '{type(other)}'")
        
        return Factor(mul_data, mul_expr)
    

    def __rmul__(self, other):

        if isinstance(other, (int, float)):
            mul_data = other * self.data
            mul_expr = f"({other} * {self.expr})"
        
        elif isinstance(other, Factor):
            mul_data = other.data * self.data
            mul_expr = f"({other.expr} * {self.expr})"
        
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(other)}' and 'Factor'")
        
        return Factor(mul_data, mul_expr)
    
    
    def __truediv__(self, other):

        if isinstance(other, (int, float)):
            div_data = self.data / other
            div_expr = f"({self.expr} / {other})"

        elif isinstance(other, Factor):
            div_data = self.data / other.data
            div_expr = f"({self.expr} / {other.expr})"
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Factor' and '{type(other)}'")
        
        return Factor(div_data, div_expr)
    

    def __rtruediv__(self, other):

        if isinstance(other, (int, float)):
            div_data = other / self.data
            div_expr = f"({other} / {self.expr})"
        
        elif isinstance(other, Factor):
            div_data = other.data / self.data
            div_expr = f"({other.expr} / {self.expr})"
        
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other)}' and 'Factor'")
        
        return Factor(div_data, div_expr)
    
    def sqrt(self):
        from .operators import sqrt
        return sqrt(self)
    
    def exp(self):
        from .operators import exp
        return exp(self)
    
    def log(self, base=None):
        from .operators import log
        return log(self, base)
    
    def power(self, exponent):
        from .operators import power
        return power(self, exponent)
    
    def signedpower(self, power):
        from .operators import signedpower
        return signedpower(self, power)
    
    def lt(self, other):
        from .operators import lt
        return lt(self, other)
    
    def gt(self, other):
        from .operators import gt
        return gt(self, other)
    
    def lte(self, other):
        from .operators import lte
        return lte(self, other)
    
    def gte(self, other):
        from .operators import gte
        return gte(self, other)
    
    def eq(self, other):
        from .operators import eq
        return eq(self, other)
    
    def ne(self, other):
        from .operators import ne
        return ne(self, other)
    
    def neg(self):
        from .operators import mul
        return mul(self, -1)
    
    def add(self, other):
        from .operators import add
        return add(self, other)
    
    def sub(self, other):
        from .operators import sub
        return sub(self, other)
    
    def mul(self, other):
        from .operators import mul
        return mul(self, other)
    
    def div(self, other):
        from .operators import div
        return div(self, other)
    
    def reverse_index(self):
        from .operators import reverse_index
        return reverse_index(self)
    
    def reindex(self, target_index):
        from .operators import reindex
        return reindex(self, target_index)
    
    def fillna(self, value=None):
        from .operators import fillna
        return fillna(self, value)
    
    def ffill(self):
        from .operators import ffill
        return ffill(self)
    
    def bfill(self):
        from .operators import bfill
        return bfill(self)
    
    def copy(self, fill_value=None):
        from .operators import copy
        return copy(self, fill_value)
    
    def delay(self, window):
        from .operators import delay
        return delay(self, window)
    
    def shift(self, periods):
        from .operators import delay
        return delay(self, periods)
    
    def delta(self, window):
        from .operators import delta
        return delta(self, window)
    
    def diff(self, periods=1):
        from .operators import delta
        return delta(self, periods)
    
    def pct_change(self, periods=1):
        from .operators import delay, div, sub
        current = self
        previous = delay(self, periods)
        return div(sub(current, previous), previous)
    
    def ewma(self, window, half_life=None):
        from .operators import ewma
        return ewma(self, window, half_life)
    
    def ts_rank(self, window):
        from .operators import ts_rank
        return ts_rank(self, window)
    
    def ts_min(self, window):
        from .operators import ts_min
        return ts_min(self, window)
    
    def ts_max(self, window):
        from .operators import ts_max
        return ts_max(self, window)
    
    def ts_argmax(self, window):
        from .operators import ts_argmax
        return ts_argmax(self, window)
    
    def ts_sum(self, window):
        from .operators import ts_sum
        return ts_sum(self, window)
    
    def ts_prod(self, window):
        from .operators import ts_prod
        return ts_prod(self, window)
    
    def ts_mean(self, window, weight=False, half_life=None):
        from .operators import ts_mean
        return ts_mean(self, window, weight, half_life)
    
    def ts_stddev(self, window, weight=False, half_life=None):
        from .operators import ts_stddev
        return ts_stddev(self, window, weight, half_life)
    
    def ts_cov(self, x, window, weight=False, half_life=None):
        from .operators import ts_cov
        return ts_cov(self, x, window, weight, half_life)
    
    def ts_corr(self, x, window, weight=False, half_life=None):
        from .operators import ts_corr
        return ts_corr(self, x, window, weight, half_life)
    
    def ts_linreg(self, x, window, param='beta', weight=False, half_life=None, compute_mode='standard'):
        from .operators import ts_linreg
        return ts_linreg(self, x, window, weight, half_life, param, compute_mode)
    
    def apply(self, func, window=None, **kwargs):
        from .operators import apply
        return apply(self, func, window, **kwargs)
    
    def resample_agg(self, freq, agg_func='mean', min_periods=0, label='right', cal_rule='calendar'):
        from .operators import resample_agg
        return resample_agg(self, freq, agg_func, min_periods, label, cal_rule)
    
    def block_agg(self, window, sections, inner_func, outer_func):
        from .operators import block_agg
        return block_agg(self, window, sections, inner_func, outer_func)
    
    def cs_rank(self):
        from .operators import cs_rank
        return cs_rank(self)
    
    def cs_linreg(self, x, weight=None, param='beta'):
        from .operators import cs_linreg
        return cs_linreg(self, x, weight, param)
    
    def normalize(self, cross_section=False, expanding=False, window=None):
        from ..utils.preprocess import normalize
        return normalize(self, cross_section, expanding=expanding, window=window)
    
    def standardize(self, cross_section=False, expanding=False, window=None):
        from ..utils.preprocess import standardize
        return standardize(self, cross_section, expanding=expanding, window=window)
    
    def clip(self, multiplier=3.0, cross_section=True, expanding=False, window=None):
        from ..utils.preprocess import clip_mad
        return clip_mad(self, multiplier, cross_section, expanding=expanding, window=window)
    
    def winsorize(self, limits=(0.1, 0.1), cross_section=True, expanding=False, window=None):
        from ..utils.preprocess import winsorize_quantile
        return winsorize_quantile(self, limits, cross_section, expanding=expanding, window=window)
    
    def industry_neutralize(
            self,
            industry_mapping=None,
            method="demean",
            classification="sw_l1",
            database_type=DatabaseTypes.DUCKDB,
            calendar_type=CalendarTypes.NONE,
        ):
        
        from ..utils.preprocess import industry_neutralize
        
        if industry_mapping is None:
            from ...utils.industry import get_industry_manager
            industry_manager = get_industry_manager(database_type)
            
            # get tickers and date range
            tickers = list(self.data.columns)
            start_date = self.data.index[0].strftime('%Y%m%d')
            end_date = self.data.index[-1].strftime('%Y%m%d')
            
            # get industry mapping
            industry_mapping = industry_manager.get_industry_time_series(
                universe=tickers,
                start_date=start_date,
                end_date=end_date,
                classification=classification,
                calendar_type=calendar_type,
                multiindex=False
            )
        
        return industry_neutralize(self, industry_mapping=industry_mapping, method=method, multiindex=False)


def factor(data_source, name):
    """
    Create a factor object from data source
    
    Parameters:
    -----------
    data_source : pd.DataFrame
        Raw data for the factor
    name : str
        Name/expression for the factor
        
    Returns:
    --------
    Factor: Factor object with data and expression
    """
    return Factor(data_source, name)

