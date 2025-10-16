from scipy.optimize import fsolve
from .execute import execute
from .constants import TransactionType

class Solver:

    def _account_func(self, shares, init_cash, buy_price, slippage_rate, slippage_type, commission_point, min_cost):

        buy_price_with_slippage = execute.apply_slippage(buy_price, slippage_rate, slippage_type, transaction_type=TransactionType.BUY)
        account_value = shares * buy_price_with_slippage
        transac_cost = account_value * commission_point
        transac_cost = transac_cost if transac_cost >= min_cost else min_cost
        
        return account_value + transac_cost - init_cash
    

    def share_solver(self, init_cash, buy_price, slippage_rate, slippage_type, commission_point, min_cost, trade_unit):

        args = (init_cash, buy_price, slippage_rate, slippage_type, commission_point, min_cost)
        init_guess = init_cash / buy_price
        shares = fsolve(self._account_func, x0=init_guess, args=args)[0]
        shares = 0 if shares < 1 else shares
        shares = int(shares / trade_unit) * trade_unit

        return shares

solver = Solver()
