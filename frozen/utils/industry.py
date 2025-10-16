import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, TYPE_CHECKING

from ..utils.log import GL
from ..utils.calendar import Calendar, CalendarTypes

if TYPE_CHECKING:
    from ..data.database import DatabaseTypes

logger = GL.get_logger("industry")


class IndustryManager:
    """
    Industry Information Manager
    
    Provides functionality for fetching, caching, and managing stock industry classification information.
    Supports multiple industry classification standards (SHENWAN, CITIC, etc.) and data formats.
    """
    
    def __init__(self, database_type: "DatabaseTypes", enable_cache: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize the industry information manager
        
        Parameters:
        -----------
        database_type : DatabaseTypes
        enable_cache : bool, default False
            Whether to enable caching mechanism
        cache_dir : str, optional
            Cache directory path, only effective when enable_cache is True, defaults to industry_cache in current directory
        """
        from ..data.etl.dataload import DataLoadManager
        self.dataloader = DataLoadManager(database_type)
        self.enable_cache = enable_cache
        
        # Only set cache directory when cache is enabled
        self.cache_dir = None
        if self.enable_cache:
            if cache_dir is None:
                cache_dir = Path.cwd() / "industry_cache"
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        
        # Industry classification mapping cache
        self._industry_mapping = {}
        
    def get_stock_industry_mapping(self, classification: str = "sw_l1", refresh: bool = False) -> pd.DataFrame:
        """
        Get stock industry mapping information
        
        Parameters:
        -----------
        classification : str
            Industry classification standard:
            - 'basic': Basic industry classification
            - 'sw_l1': SHENWAN Level 1 industry
            - 'sw_l2': SHENWAN Level 2 industry
            - 'sw_l3': SHENWAN Level 3 industry
            - 'zx_l1': CITIC Level 1 industry
            - 'zx_l2': CITIC Level 2 industry
            - 'zx_l3': CITIC Level 3 industry
            - 'custom': Custom industry classification
        refresh : bool
            Whether to force refresh cache
            
        Returns:
        --------
        pd.DataFrame: DataFrame containing ticker and industry columns
        """
        
        # Handle custom classification from memory
        if classification == "custom":
            if classification in self._industry_mapping and self._industry_mapping[classification]:
                custom_data = []
                for ticker, industry in self._industry_mapping[classification].items():
                    custom_data.append({"ticker": ticker, "industry": industry})
                return pd.DataFrame(custom_data)
            else:
                logger.warning("No custom industry classifications found")
                return pd.DataFrame(columns=["ticker", "industry"])
        
        if classification not in ["basic", "sw_l1", "sw_l2", "sw_l3", "zx_l1", "zx_l2", "zx_l3"]:
            raise ValueError(f"Unsupported classification: {classification}")
        
        # Check cache (only when cache is enabled)
        if self.enable_cache and not refresh:
            cache_file = self.cache_dir / f"industry_mapping_{classification}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                        # Check if cache is expired (1 day)
                        if datetime.now() - cached_data["timestamp"] < timedelta(days=1):
                            logger.info(f"Using cached industry mapping for {classification}")
                            # Apply custom overrides to cached data (don't modify cache)
                            return self._apply_custom_overrides(cached_data["data"], classification)
                except Exception as e:
                    logger.warning(f"Failed to load cached data: {e}")
        
        # Fetch new data
        logger.info(f"Fetching industry mapping for {classification}")
        industry_data = self.dataloader.load_industry_mapping(classification)
        
        # Cache original data (only when cache is enabled) - cache BEFORE applying overrides
        if self.enable_cache:
            cache_file = self.cache_dir / f"industry_mapping_{classification}.pkl"
            cache_data = {
                "data": industry_data.copy(),  # Cache original data
                "timestamp": datetime.now()
            }
            
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Cached industry mapping for {classification}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        # Apply custom overrides to the result (after caching original data)
        industry_data = self._apply_custom_overrides(industry_data, classification)
        
        return industry_data
    
    def _apply_custom_overrides(self, industry_data: pd.DataFrame, classification: str) -> pd.DataFrame:
        """
        Apply custom industry classification overrides to the base data
        
        Parameters:
        -----------
        industry_data : pd.DataFrame
            Base industry mapping data
        classification : str
            Classification standard name
            
        Returns:
        --------
        pd.DataFrame: Industry data with custom overrides applied
        """
        if classification not in self._industry_mapping or not self._industry_mapping[classification]:
            return industry_data
        
        # Make a copy to avoid modifying the original data
        result_data = industry_data.copy()
        
        # Apply custom overrides
        custom_mapping = self._industry_mapping[classification]
        for ticker, custom_industry in custom_mapping.items():
            # Update existing ticker or add new one
            if len(result_data) > 0:
                mask = result_data["ticker"] == ticker
                if mask.any():
                    result_data.loc[mask, "industry"] = custom_industry
                    logger.debug(f"Override industry for {ticker}: {custom_industry}")
                else:
                    # Add new ticker with custom industry
                    new_row = pd.DataFrame({"ticker": [ticker], "industry": [custom_industry]})
                    result_data = pd.concat([result_data, new_row], ignore_index=True)
                    logger.debug(f"Added new ticker {ticker} with industry: {custom_industry}")
            else:
                # If result_data is empty, just add the custom mapping
                new_row = pd.DataFrame({"ticker": [ticker], "industry": [custom_industry]})
                result_data = pd.concat([result_data, new_row], ignore_index=True)
                logger.debug(f"Added new ticker {ticker} with industry: {custom_industry}")
        
        return result_data
    
    def get_industry_time_series(
            self,
            universe: List[str],
            start_date: str,
            end_date: str,
            classification: str = "basic",
            calendar_type: CalendarTypes = CalendarTypes.NONE,
            multiindex: bool = False
        ) -> pd.DataFrame:
        """
        Get industry information in time series format
        
        Parameters:
        -----------
        universe : List[str]
            List of stock tickers
        start_date : str
        end_date : str
        classification : str
            Industry classification standard
            
        Returns:
        --------
        pd.DataFrame: index is date, columns are stock tickers, values are industry classifications. If multiindex is True, index is (ticker, trade_date), column is industry classification.
        """
        
        # Get industry mapping
        industry_mapping = self.get_stock_industry_mapping(classification)
        
        # Create ticker to industry mapping dictionary
        ticker_to_industry = dict(zip(industry_mapping["ticker"], industry_mapping["industry"]))
        
        # Generate date range
        calendar = Calendar(calendar_type)
        date_range = calendar.generate(start_date, end_date)
        
        if not multiindex:
            # Create time series DataFrame
            industry_ts = pd.DataFrame(index=date_range, columns=universe)
            
            # Fill industry information (assuming industry classification doesn't change within the time period)
            for ticker in universe:
                if ticker in ticker_to_industry:
                    industry_ts[ticker] = ticker_to_industry[ticker]
                else:
                    # Use default value if industry information is not found
                    industry_ts[ticker] = "未分类"
                    logger.warning(f"No industry information found for {ticker}")
            
            industry_ts.index.name = "trade_date"

            #########################################################
            # Vectorized approach 1:
            # # 1. Create a Series with industry for each ticker
            # industry_series = pd.Series(
            #     [ticker_to_industry.get(ticker, "未分类") for ticker in universe],
            #     index=universe
            # )
            
            # # 2. Create a DataFrame where each row is the industry series
            # #    (repeated for each date in the range)
            # industry_ts = pd.concat(
            #     [industry_series] * len(date_range),
            #     axis=1
            # ).T
            
            # # Set index to date range
            # industry_ts.index = date_range
            
            # # Log warnings for missing tickers
            # missing_tickers = [ticker for ticker in universe if ticker not in ticker_to_industry]
            # if missing_tickers:
            #     logger.warning(f"No industry information found for {len(missing_tickers)} tickers: "
            #                 f"{', '.join(missing_tickers[:10])}{'...' if len(missing_tickers) > 10 else ''}")

            #########################################################
            # Vectorized approach 2:
            # industries = np.array([ticker_to_industry.get(ticker, "Unclassified") for ticker in universe])
            # industry_ts = pd.DataFrame(
            #     np.tile(industries, (len(date_range), 1)),
            #     index=date_range,
            #     columns=universe
            # )
            
            industry_ts = industry_ts.sort_index(axis=0).sort_index(axis=1)
            return industry_ts
        else:
            industry_list = []
            
            for date in date_range:
                for ticker in universe:
                    industry = ticker_to_industry.get(ticker, "未分类")
                    industry_list.append({
                        "trade_date": date,
                        "ticker": ticker,
                        "industry": industry
                    })
            
            industry_ts = pd.DataFrame(industry_list)
            industry_ts = industry_ts.set_index(["ticker", "trade_date"]).sort_index()
            return industry_ts
    
    def update_industry_classification(self, 
                                     ticker: str, 
                                     industry: str, 
                                     classification: str = "custom"):
        """
        Update industry classification for specific stock
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        industry : str
            New industry classification
        classification : str
            Classification standard name. Use 'custom' for pure custom classification,
            or existing standard names ('basic', 'sw_l1', etc.) to override specific stocks
        """
        
        if classification not in self._industry_mapping:
            self._industry_mapping[classification] = {}
        
        self._industry_mapping[classification][ticker] = industry
        logger.info(f"Updated industry classification for {ticker}: {industry} (classification: {classification})")
    
    def remove_industry_classification(self, ticker: str, classification: str = "custom"):
        """
        Remove custom industry classification for specific stock
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        classification : str
            Classification standard name
        """
        if classification in self._industry_mapping and ticker in self._industry_mapping[classification]:
            del self._industry_mapping[classification][ticker]
            logger.info(f"Removed custom industry classification for {ticker} (classification: {classification})")
        else:
            logger.warning(f"No custom classification found for {ticker} in {classification}")
    
    def get_custom_classifications(self, classification: str = "custom") -> Dict[str, str]:
        """
        Get all custom industry classifications for a specific standard
        
        Parameters:
        -----------
        classification : str
            Classification standard name
            
        Returns:
        --------
        Dict[str, str]: Mapping from ticker to industry
        """
        return self._industry_mapping.get(classification, {}).copy()
    
    def get_industry_statistics(self, classification: str = "basic") -> Dict:
        """
        Get industry classification statistics
        
        Returns:
        --------
        Dict: Contains number of industries, number of stocks per industry and other statistical information
        """
        
        industry_mapping = self.get_stock_industry_mapping(classification)
        
        stats = {
            "total_stocks": len(industry_mapping),
            "total_industries": industry_mapping["industry"].nunique(),
            "industry_counts": industry_mapping["industry"].value_counts().to_dict(),
            "missing_industry_count": industry_mapping["industry"].isna().sum()
        }
        
        return stats
    
    def validate_industry_coverage(self, universe: List[str], classification: str = "basic") -> Dict:
        """
        Validate industry coverage of stock universe
        
        Returns:
        --------
        Dict: Contains coverage rate, missing stock list and other information
        """
        
        industry_mapping = self.get_stock_industry_mapping(classification)
        available_tickers = set(industry_mapping["ticker"])
        universe_set = set(universe)
        
        covered_tickers = universe_set.intersection(available_tickers)
        missing_tickers = universe_set - available_tickers
        
        coverage = {
            "total_stocks": len(universe),
            "covered_stocks": len(covered_tickers),
            "missing_stocks": len(missing_tickers),
            "coverage_rate": len(covered_tickers) / len(universe) if universe else 0,
            "missing_tickers": list(missing_tickers)
        }
        
        return coverage
    
    def set_cache_enabled(self, enabled: bool, cache_dir: Optional[str] = None):
        """
        Dynamically set cache enabled status
        
        Parameters:
        -----------
        enabled : bool
            Whether to enable cache
        cache_dir : str, optional
            Cache directory path, only effective when enabled is True
        """
        self.enable_cache = enabled
        if enabled:
            if self.cache_dir is None:
                if cache_dir is None:
                    cache_dir = Path.cwd() / "industry_cache"
                self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            logger.info("Industry cache enabled")
        else:
            logger.info("Industry cache disabled")
    
    def clear_cache(self):
        """Clear all cache files"""
        if not self.enable_cache or self.cache_dir is None:
            logger.warning("Cache is disabled or cache directory not set, nothing to clear")
            return
            
        cache_files = list(self.cache_dir.glob("industry_mapping_*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")


# Global industry manager instance
_global_industry_manager = None

def get_industry_manager(database_type: Optional["DatabaseTypes"] = None, enable_cache: bool = False, cache_dir: Optional[str] = None) -> IndustryManager:
    """
    Get global industry manager instance
    
    Parameters:
    -----------
    database_type : DatabaseTypes, optional
        Database type, must be provided when called for the first time
    enable_cache : bool, default False
        Whether to enable caching mechanism
    cache_dir : str, optional
        Cache directory path, only effective when enable_cache is True
    """
    global _global_industry_manager
    if _global_industry_manager is None:
        if database_type is None:
            raise ValueError("database_type must be provided when creating industry manager for the first time")
        _global_industry_manager = IndustryManager(database_type, enable_cache=enable_cache, cache_dir=cache_dir)
    return _global_industry_manager 