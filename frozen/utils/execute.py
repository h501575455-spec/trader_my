import numpy as np
import pandas as pd
from typing import Union, Tuple
from .constants import SlippageType, TransactionType

class Execute:
    """Order execution module"""

    def __init__(self):
        pass

    def apply_slippage(
            self,
            price: Union[float, np.array],
            slippage: float = 0.00,
            slippage_type: Union[str, SlippageType] = SlippageType.PERCENTAGE,
            transaction_type: Union[str, TransactionType] = TransactionType.BUY
        ):
        """
        Apply the rate and type of slippage on the original price during transaction.
        
        Parameters:
        -----------
        price: Union[float, np.array]
            The original instrument price to buy or sell.
        
        slippage: float
            The slippage point to apply, should be adjusted across different markets.

        slippage_type: Union[str, SlippageType]
            The type of slippage to apply, currently support two types.
            - PERCENTAGE: Apply the percentage point based on the original price.
            - ABSOLUTE: Apply the absolute value directly on the original price.
        
        transaction_type: Union[str, TransactionType]
            The trade direction when placing orders.
            - BUY: Long instrument position.
            - SELL: Short instrument position.
        """
        
        slippage_type = SlippageType(slippage_type) if isinstance(slippage_type, str) else slippage_type
        transaction_type = TransactionType(transaction_type) if isinstance(transaction_type, str) else transaction_type

        if slippage_type == SlippageType.PERCENTAGE:
            if transaction_type == TransactionType.BUY:
                price_with_slippage = price * (1 + slippage)
            elif transaction_type == TransactionType.SELL:
                price_with_slippage = price * (1 - slippage)
            else:
                raise ValueError("Unsupported transaction type! Please check the available transaction type.")
        elif slippage_type == SlippageType.ABSOLUTE:
            if transaction_type == TransactionType.BUY:
                price_with_slippage = price + slippage
            elif transaction_type == TransactionType.SELL:
                price_with_slippage = price - slippage
            else:
                raise ValueError("Unsupported transaction type! Please check the available transaction type.")
        else:
            raise ValueError("Unsupported slippage type! Please indicate the type of slippage to use.")
        
        return price_with_slippage
    

    def inst_trade_status(
            self,
            instrument: str,
            date: pd.Timestamp,
            open_price: str,
            up_limit: pd.DataFrame,
            down_limit: pd.DataFrame,
            suspend: pd.DataFrame,
            direction: TransactionType
        ) -> Tuple[bool]:

        suspend_status, limit_status = False, False
        
        # Check suspend status
        suspend_all = suspend.loc[date]
        suspend_all = suspend_all.to_frame().T if isinstance(suspend_all, pd.Series) else suspend_all
        if instrument in suspend_all["ticker"].tolist():
            if suspend_all[suspend_all["ticker"]==instrument]["suspend_timing"].values[0] is None:
                suspend_status = True
        
        # Check limit status
        if direction == TransactionType.BUY:
            try:
                if round(open_price, 2) == up_limit.loc[date, instrument]:
                    limit_status = True
            except:
                limit_status = False
            buy_status = not (suspend_status or limit_status)
            return buy_status, suspend_status, limit_status
    
        elif direction == TransactionType.SELL:
            try:
                if round(open_price, 2) == down_limit.loc[date, instrument]:
                    limit_status = True
            except:
                limit_status = False
            sell_status = not (suspend_status or limit_status)
            return sell_status, suspend_status, limit_status
        else:
            raise ValueError("Invalid trade direction!")

execute = Execute()
