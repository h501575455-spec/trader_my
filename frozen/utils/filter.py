import re
import pandas as pd
from typing import List, Tuple, TYPE_CHECKING

from ..data.provider import TickerType, ProviderTypes

if TYPE_CHECKING:
    from ..basis.constant import FrozenConfig


def prelim_pool_screen(config: "FrozenConfig", universe: List[str]) -> List[str]:
    """
    Pipeline designed to filter out stocks that meet specific 
    criteria based on the configuration.
    
    Parameters
    ----------
    config : FrozenConfig
        Configuration object containing screening criteria
        
    Returns
    -------
    List[str]
        List of screened stock tickers
    """
    from ..data.etl.dataload import DataLoadManager
    dataloader = DataLoadManager(config)
    
    # get all listed stocks
    all_stocks = dataloader.load_basic_info("stock_basic")

    # initialize exclude list
    exclude_set = set()

    # exclude ST stocks
    if not config.include_ST:
        st_ticker = set(all_stocks[all_stocks.name.str.contains("ST", na=False)]["ticker"])
        exclude_set = exclude_set.union(st_ticker)

    # exclude SH stocks
    if not config.include_SH:
        sh_ticker = set(all_stocks[all_stocks.ticker.str.contains("^60", na=False)]["ticker"])
        exclude_set = exclude_set.union(sh_ticker)

    # exclude SZ stocks
    if not config.include_SZ:
        sz_ticker = set(all_stocks[all_stocks.ticker.str.contains("^00", na=False)]["ticker"])
        exclude_set = exclude_set.union(sz_ticker)

    # exclude BJ stocks
    if not config.include_BJ:
        bj_ticker = set(all_stocks[all_stocks.ticker.str.contains("^8", na=False)]["ticker"])
        exclude_set = exclude_set.union(bj_ticker)

    # exclude GEM stocks
    if not config.include_GEM:
        gem_ticker = set(all_stocks[all_stocks.ticker.str.contains("^30", na=False)]["ticker"])
        exclude_set = exclude_set.union(gem_ticker)

    # exclude STAR stocks
    if not config.include_STAR:
        star_ticker = set(all_stocks[all_stocks.ticker.str.contains("^68", na=False)]["ticker"])
        exclude_set = exclude_set.union(star_ticker)

    # exclude stocks with less than min_list_days year of listing
    list_threshold_date = config.calendar.adjust(config.start_date, -config.min_list_days)
    list_data = set(all_stocks[all_stocks.list_date > list_threshold_date]["ticker"])
    exclude_set = exclude_set.union(list_data)

    # exclude delisted stocks
    if not config.include_delist:
        delist_stocks = all_stocks[all_stocks.list_status == "D"]
        delist_ticker = set(delist_stocks.ticker)
        exclude_set = exclude_set.union(delist_ticker)

    # get screened ticker
    screened_ticker = set(universe).difference(exclude_set)

    return list(screened_ticker)


class Universe:
    """
    Stock pool management class, responsible for building and maintaining the stock pool
    
    Examples
    --------
    # use default config to build stock pool
    universe = Universe(config)
    tickers = universe.pool
    
    # use custom initial stock pool
    custom_pool = ['000001.SZ', '000002.SZ', '600000.SH']
    universe = Universe(config, initial_pool=custom_pool)
    filtered_tickers = universe.pool
    
    # dynamic update initial stock pool
    new_pool = ['000001.SZ', '600036.SH', '000858.SZ']
    updated_tickers = universe.update_initial_pool(new_pool)
    """
    
    def __init__(self, config: "FrozenConfig", initial_pool: List[str] = None):
        """
        Parameters
        ----------
        config : FrozenConfig
            Frozen configuration object
        initial_pool : List[str], optional
            custom initial stock pool, if provided, use it to screen,
            otherwise use the stock pool according to the configuration
        """
        self.config = config
        self.initial_pool = initial_pool
        self._pool = None  # cache stock pool

        # initialize dataloader
        from ..data.etl.dataload import DataLoadManager
        self.dataloader = DataLoadManager(self.config)
    
    @property
    def pool(self) -> Tuple[str, ...]:
        """get final screened pool (tuple)"""
        if self._pool is None:
            self._build_universe()
        return tuple(self._pool)
    
    def refresh(self) -> Tuple[str, ...]:
        """refresh pool (when need to be updated)"""
        self._pool = None
        return self.pool
    
    def update_initial_pool(self, new_pool: List[str]) -> Tuple[str, ...]:
        """
        update initial stock pool and re-screen
        
        Parameters
        ----------
        new_pool : List[str]
            new initial stock pool
            
        Returns
        -------
        Tuple[str, ...]
            screened stock pool
        """
        self.initial_pool = new_pool
        self._pool = None
        return self.pool
    
    def _build_universe(self):
        """build stock pool: get initial pool + screen"""
        from ..data.provider.factory import ProviderFactory
        # step 1: get initial stock pool
        if self.initial_pool is not None:
            # if custom initial stock pool is provided, use it directly
            raw_pool = self.initial_pool
        elif hasattr(self.config, "index_code") and self.config.index_code:
            # if index code is provided, get index constituents
            try:
                raw_pool = self.dataloader.loade_index_weight(
                    index_code=self.config.index_code,
                    target_date=self.config.start_date,
                    code_only=True
                )
            except:
                provider = ProviderFactory.create_data_feed(ProviderTypes.TUSHARE)
                raw_pool = provider.get_instrument_list(
                    ticker_type=TickerType.INDEX_COMPONENT,
                    index_code=self.config.index_code,
                    start_date=self.config.start_date
                )
        else:
            # get all listed stocks by default
            try:
                raw_pool = self.dataloader.load_basic_info("stock_basic")
                raw_pool = raw_pool.ticker.tolist()
            except:
                provider = ProviderFactory.create_data_feed(ProviderTypes.TUSHARE)
                raw_pool = provider.get_instrument_list(
                    ticker_type=TickerType.LISTED_STOCK
                )
        
        # step 2: execute screening logic
        self._pool = self._screen_pool(raw_pool)
    
    def _screen_pool(self, initial_pool: List[str]) -> List[str]:
        """execute screening logic"""
        # load all listed stocks
        all_stocks = self.dataloader.load_basic_info("stock_basic")
        
        # initialize exclude set
        exclude_set = set()
        
        # exclude ST stocks
        if not self.config.include_ST:
            st_mask = all_stocks["name"].str.contains("ST")
            exclude_set.update(all_stocks[st_mask]["ticker"])

        # exclude SH stocks
        if not self.config.include_SH:
            sh_mask = all_stocks["ticker"].str.startswith("60")
            exclude_set.update(all_stocks[sh_mask]["ticker"])
        
        if not self.config.include_SZ:
            sz_mask = all_stocks["ticker"].str.startswith("00")
            exclude_set.update(all_stocks[sz_mask]["ticker"])
        
        if not self.config.include_BJ:
            bj_mask = all_stocks["ticker"].str.startswith("8")
            exclude_set.update(all_stocks[bj_mask]["ticker"])

        # exclude GEM stocks
        if not self.config.include_GEM:
            gem_mask = all_stocks["ticker"].str.startswith("30")
            exclude_set.update(all_stocks[gem_mask]["ticker"])

        # exclude STAR stocks
        if not self.config.include_STAR:
            star_mask = all_stocks["ticker"].str.startswith("68")
            exclude_set.update(all_stocks[star_mask]["ticker"])

        # exclude stocks with less than min_list_days year of listing
        list_threshold_date = self.config.calendar.adjust(self.config.start_date, -self.config.min_list_days)
        list_data = set(all_stocks[all_stocks.list_date > list_threshold_date]["ticker"])
        exclude_set.update(list_data)

        # exclude delisted stocks
        if not self.config.include_delist:
            delist_stocks = all_stocks[all_stocks.list_status == "D"]
            delist_ticker = set(delist_stocks.ticker)
            exclude_set.update(delist_ticker)

        # Apply condition-based screening if conditions are defined
        remaining_pool = list(set(initial_pool).difference(exclude_set))
        if self.config.filter_condition:
            remaining_pool = self._apply_condition_filter(remaining_pool)

        # get screened ticker
        screened_ticker = remaining_pool

        return screened_ticker
    
    def _apply_condition_filter(self, pool: List[str]) -> List[str]:
        """Apply condition-based filtering to the stock pool"""
        if not pool:
            return pool
        
        conditions = self.config.filter_condition
        if not conditions:
            return pool
        
        # Get target date from config, fallback to start date
        target_date = self.config.filter_target_date or self.config.start_date

        # Identify required columns
        required_col = list(conditions.keys())

        # Get database table columns
        fundamental_columns = self.dataloader.load_column_schema("stock_daily_fundamental")
        price_columns = self.dataloader.load_column_schema("stock_daily_real")
        fundamental_columns = fundamental_columns["column_name"].tolist()
        price_columns = price_columns["column_name"].tolist()

        # Get common columns
        required_fundamental_col = tuple(set(required_col).intersection(set(fundamental_columns)))
        required_price_col = tuple(set(required_col).intersection(set(price_columns)))

        try:
            # Load fundamental data
            fundamental_data = self.dataloader.load_volume_price(
                "stock_daily_fundamental",
                required_fundamental_col,
                universe=tuple(pool),
                start_date=target_date,
                end_date=target_date,
                multiindex=True
            )
            
            # Load price data
            price_data = self.dataloader.load_volume_price(
                "stock_daily_real",
                required_price_col,
                universe=tuple(pool),
                start_date=target_date,
                end_date=target_date,
                multiindex=True
            )
            
            combined_data = pd.DataFrame(index=pool)
            
            if fundamental_data:
                fund_df = fundamental_data.xs(target_date, level="trade_date", drop_level=False).reset_index(level="trade_date", drop=True)
                combined_data = combined_data.join(fund_df)
            
            if price_data is not None:
                price_df = price_data.xs(target_date, level="trade_date", drop_level=False).reset_index(level="trade_date", drop=True)
                new_cols = price_df.columns.difference(combined_data.columns)
                if not new_cols.empty:
                    combined_data = combined_data.join(price_df[new_cols])
            
            # Apply condition filters
            filtered_tickers = self._evaluate_condition(combined_data, pool)
            return filtered_tickers
        
        except Exception as e:
            # If condition filtering fails, return original pool
            print(f"Warning: Condition filtering failed: {e}")
            return pool
    
    def _evaluate_condition(self, all_data: pd.DataFrame, pool: List[str]) -> List:
        """High performance evaluation of conditions"""
        # Build query string from conditions
        query_parts = []
        for field, condition in self.config.filter_condition.items():
            if field in all_data.columns:
                # Normalize condition string
                cond = condition.strip().replace(" ", "")
                # Handle compound conditions
                if "and" in cond.lower() or "or" in cond.lower():
                    # Split compound conditions
                    parts = re.split(r"(and|or)", cond, flags=re.IGNORECASE)
                    processed = []
                    for part in parts:
                        if part.lower() in ["and", "or"]:
                            processed.append(part.lower())
                        else:
                            processed.append(f"`{field}`{part}")
                    query_parts.append(" ".join(processed))
                else:
                    query_parts.append(f"`{field}`{cond}")
        
        # Execute the query if we have valid conditions
        if query_parts:
            full_query = " and ".join(query_parts)
            
            # Handle special cases like NaN and None
            full_query = full_query.replace("!=nan", ".notna()")
            full_query = full_query.replace("==nan", ".isna()")
            
            try:
                # Execute the query
                filtered_df = all_data.query(full_query, engine="python")
                filtered_ticker = filtered_df.index.tolist()
                return filtered_ticker
            except Exception as e:
                print(f"Query failed: {full_query}. Error: {str(e)}")
        
        # Fallback to original pool if no valid conditions
        return pool


# def preliminary_screening(pool: pd.DataFrame, delist: pd.DataFrame, universe: list) -> list:
#     """
#     Pipeline designed to filter out stocks that meet specific criteria.
#     """

#     # remove ST stocks
#     st_data = set(pool[pool.name.str.contains("ST")]["ts_code"])
#     # remove stocks from 'Growth Enterprise Market (GEM)' and 'Science and Technology Innovation Board (STAR Market)'
#     cyks_data = set(pool[pool.ts_code.str.contains("^30|^68")]["ts_code"])
#     # exclude stocks with less than one year of listing
#     list_threshold = (datetime.strptime(frozen_config.start_date, "%Y%m%d") + relativedelta(years=-1)).strftime("%Y%m%d")
#     list_data = set(pool[pool.list_date > list_threshold]["ts_code"])
#     # exclude stocks that have been delisted
#     delist_data = set(delist.ts_code)

#     exclude_list = delist_data.union(list_data.union(st_data.union(cyks_data)))
#     screened_list = list(set(universe).difference(exclude_list))

#     return screened_list

