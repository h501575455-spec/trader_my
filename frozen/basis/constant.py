import os
import collections
import numpy as np
from .yamler import yamler
from ..utils.calendar import Calendar, CalendarTypes
from ..utils.helper import unpack_dict_for_init
from ..utils.validate import validate_parameters

class FrozenConfig:
    
    @staticmethod
    def tracked_property(section, config_key, triggers_calc=False):
        """Property decorator for automatically syncing attributes to all_config"""
        def decorator(func):
            name = func.__name__
            private_name = f"_{name}"
            
            def getter(self):
                return getattr(self, private_name)
            
            def setter(self, value):
                setattr(self, private_name, value)
                # Update all_config
                if hasattr(self, "all_config"):
                    self._set_nested_config(f"{section}.{config_key}", value)
                # Trigger calc if needed
                if triggers_calc and hasattr(self, "_initialized") and self._initialized:
                    self._calc_extra_params()
            
            return property(getter, setter)
        return decorator
    
    def _set_nested_config(self, path_str, value):
        """Set value in nested dict using dot notation path"""
        keys = path_str.split(".")
        config = self.all_config
        
        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
        
        # Sync back to properties if they exist
        self._sync_nested_to_properties(path_str, value)
    
    def _sync_nested_to_properties(self, path_str, value):
        """Sync nested config changes back to property private variables"""
        # Define mapping from nested paths to property attributes
        path_to_property = {
            "basic_config.database": "database",

            "strategy_config.benchmark": "benchmark",
            "strategy_config.index_code": "index_code",
            "strategy_config.start_date": "start_date",
            "strategy_config.end_date": "end_date",
            "strategy_config.asset_range": "asset_range",
            "strategy_config.date_rule": "date_rule",
            
            "run_config.engine": "engine",
            "run_config.region": "region",
            "run_config.trade_unit": "trade_unit",
            "run_config.limit_threshold": "limit_threshold",
            "run_config.sell_rule": "sell_rule",
            "run_config.seed": "seed",
            "run_config.enable_gif": "enable_gif",
            
            "client_config.init_capital": "init_capital",
            "client_config.slippage.slippage_type": "slippage_type",
            "client_config.slippage.slippage_rate": "slippage_rate",
            "client_config.commission": "commission",
            "client_config.stamp_duty": "stamp_duty",
            "client_config.min_cost": "min_cost",
            
            "portfolio_config.optimizer": "optimizer",
            "portfolio_config.cov_est.cov_method": "cov_method",
            "portfolio_config.cov_est.cov_window": "cov_window",
            "portfolio_config.opt_func": "opt_func",
            "portfolio_config.long_short": "long_short",

            "risk_config.take_profit": "take_profit",
            "risk_config.stop_loss": "stop_loss",
            "risk_config.max_position_size": "max_position_size",
            "risk_config.sector_concentration": "sector_concentration",
            "risk_config.rebalance_threshold": "rebalance_threshold",
            
            "factor_config.calc.lookback_window": "lookback_window",
            "factor_config.filter.universe.include_SH": "include_SH",
            "factor_config.filter.universe.include_SZ": "include_SZ",
            "factor_config.filter.universe.include_BJ": "include_BJ",
            "factor_config.filter.universe.include_GEM": "include_GEM",
            "factor_config.filter.universe.include_STAR": "include_STAR",
            "factor_config.filter.universe.include_ST": "include_ST",
            "factor_config.filter.universe.include_delist": "include_delist",
            "factor_config.filter.universe.min_list_days": "min_list_days",
            
            # Filter condition mapping
            "factor_config.filter.condition": "filter_condition",
            
            # Filter target date mapping
            "factor_config.filter.target_date": "filter_target_date",
            
            # Preprocess pipeline mappings
            "factor_config.preprocess.pipeline.order": "preprocess_order",
            "factor_config.preprocess.pipeline.auto_apply": "preprocess_auto_apply",
            
            # Normalize mappings
            "factor_config.preprocess.normalize.enabled": "normalize_enabled",
            "factor_config.preprocess.normalize.cross_section": "normalize_cross_section",
            "factor_config.preprocess.normalize.expanding": "normalize_expanding",
            "factor_config.preprocess.normalize.window": "normalize_window",
            "factor_config.preprocess.normalize.auto_apply": "normalize_auto_apply",
            
            # Standardize mappings
            "factor_config.preprocess.standardize.enabled": "standardize_enabled",
            "factor_config.preprocess.standardize.cross_section": "standardize_cross_section",
            "factor_config.preprocess.standardize.expanding": "standardize_expanding",
            "factor_config.preprocess.standardize.window": "standardize_window",
            "factor_config.preprocess.standardize.auto_apply": "standardize_auto_apply",
            
            # Clip mappings
            "factor_config.preprocess.clip.enabled": "clip_enabled",
            "factor_config.preprocess.clip.expanding": "clip_expanding",
            "factor_config.preprocess.clip.window": "clip_window",
            "factor_config.preprocess.clip.multiplier": "clip_multiplier",
            "factor_config.preprocess.clip.auto_apply": "clip_auto_apply",
            
            # Winsorize mappings
            "factor_config.preprocess.winsorize.enabled": "winsorize_enabled",
            "factor_config.preprocess.winsorize.limits": "winsorize_limits",
            "factor_config.preprocess.winsorize.expanding": "winsorize_expanding",
            "factor_config.preprocess.winsorize.window": "winsorize_window",
            "factor_config.preprocess.winsorize.auto_apply": "winsorize_auto_apply",
            
            # Industry neutralize mappings
            "factor_config.preprocess.industry_neutralize.enabled": "industry_neutralize_enabled",
            "factor_config.preprocess.industry_neutralize.method": "industry_neutralize_method",
            "factor_config.preprocess.industry_neutralize.classification": "industry_neutralize_classification",
            "factor_config.preprocess.industry_neutralize.auto_apply": "industry_neutralize_auto_apply",
        }
        
        if path_str in path_to_property:
            property_name = path_to_property[path_str]
            private_name = f"_{property_name}"
            
            # Update the private variable directly (bypass property setter to avoid recursion)
            if hasattr(self, private_name):
                setattr(self, private_name, value)
                
                # Trigger calc if this property would normally trigger it
                calc_triggers = {"region", "start_date", "cov_window", "lookback_window"}
                if property_name in calc_triggers and hasattr(self, "_initialized") and self._initialized:
                    self._calc_extra_params()

    def __init__(self, calling_file=None):
        """
        Initialize configuration with optional strategy-specific config
        
        Args:
            calling_file (str): path to strategy code (e.g., "demo/strategy/momentum/momentum.py")
        """

        self.proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.strategy_path = os.path.abspath(calling_file) if calling_file else None
        self.strategy_folder = os.path.dirname(self.strategy_path) if calling_file else None
        self.strategy_name = os.path.basename(self.strategy_path).replace(".py", "") if calling_file else None
        
        # Load configurations
        self.all_config = self._load_config()
        self.validate()

        #######################################################################
        
        # Basic Config
        basic_config = self.all_config.get("base_config", {})

        self._database = basic_config.get("database", "duckdb")

        #######################################################################
        
        # Framework Config
        run_config = self.all_config.get("run_config", {})
        
        self._engine = run_config.get("engine", "frozen")
        self._region = run_config.get("region", "cn")
        self._trade_unit = run_config.get("trade_unit", 100)
        self._limit_threshold = run_config.get("limit_threshold", np.Inf)
        match self._region:
            case "cn":
                self._trade_unit = 100
                self._limit_threshold = 0.1
            case "us":
                self._trade_unit = 1
                self._limit_threshold = np.Inf
        self._sell_rule = run_config.get("sell_rule", "currentBar")
        self._seed = run_config.get("seed", 42)
        self._enable_gif = run_config.get("enable_gif", False)

        #######################################################################

        # Client Config
        client_config = self.all_config.get("client_config", {})
        slippage = client_config.get("slippage", {})

        self._init_capital = client_config.get("init_capital", 1000000)
        self._slippage_type = slippage.get("slippage_type", "percentage")
        self._slippage_rate = slippage.get("slippage_rate", 0.0001)
        self._commission = client_config.get("commission", 0.0003)
        self._stamp_duty = client_config.get("stamp_duty", 0.001)
        self._min_cost = client_config.get("min_cost", 5)

        #######################################################################
        
        # Strategy Config
        strategy_config = self.all_config.get("strategy_config", {})
        
        self._benchmark = strategy_config.get("benchmark", "000300.SH")
        self._index_code = strategy_config.get("index_code", "000300.SH")
        self._start_date = strategy_config.get("start_date", "20240101")
        self._end_date = strategy_config.get("end_date", "20250101")
        self._asset_range = strategy_config.get("asset_range", (0, 5))
        self._date_rule = strategy_config.get("date_rule", "W-MON")

        #######################################################################

        # Portfolio Config
        portfolio_config = self.all_config.get("portfolio_config", {})
        cov_est = portfolio_config.get("cov_est", {})

        self._optimizer = portfolio_config.get("optimizer", "mean-variance")
        self._cov_method = cov_est.get("cov_method", "custom")
        self._cov_window = cov_est.get("cov_window", 60)
        self._opt_func = portfolio_config.get("opt_func", "sharpe")
        self._long_short = portfolio_config.get("long_short", False)

        #######################################################################

        # Risk Config (strategy-specific)
        risk_config = self.all_config.get("risk_config", {})
        
        self._take_profit = run_config.get("take_profit", np.Inf)
        self._stop_loss = run_config.get("stop_loss", -np.Inf)
        self._max_position_size = risk_config.get("max_position_size", 0.1)
        self._sector_concentration = risk_config.get("sector_concentration", 0.3)
        self._rebalance_threshold = risk_config.get("rebalance_threshold", 0.05)

        #######################################################################

        # Factor Config
        factor_config = self.all_config.get("factor_config", {})

        calc = factor_config.get("calc", {})
        self._lookback_window = calc.get("lookback_window", 60)

        filter = factor_config.get("filter", {})
        universe = filter.get("universe", {})
        
        self._include_SH = universe.get("include_SH", True)
        self._include_SZ = universe.get("include_SZ", True)
        self._include_BJ = universe.get("include_BJ", False)
        self._include_GEM = universe.get("include_GEM", False)
        self._include_STAR = universe.get("include_STAR", False)
        self._include_ST = universe.get("include_ST", False)
        self._include_delist = universe.get("include_delist", False)
        self._min_list_days = universe.get("min_list_days", 100)

        self._filter_condition = filter.get("condition", {})
        self._filter_target_date = filter.get("target_date", None)

        # Factor Preprocess Config
        preprocess = factor_config.get("preprocess", {})
        
        # Pipeline config
        pipeline = preprocess.get("pipeline", {})
        self._preprocess_order = pipeline.get("order", [])
        self._preprocess_auto_apply = pipeline.get("auto_apply", False)
        
        # Normalize config
        normalize = preprocess.get("normalize", {})
        self._normalize_enabled = normalize.get("enabled", False)
        self._normalize_cross_section = normalize.get("cross_section", False)
        self._normalize_expanding = normalize.get("expanding", False)
        self._normalize_window = normalize.get("window", None)
        self._normalize_auto_apply = normalize.get("auto_apply", False)
        
        # Standardize config
        standardize = preprocess.get("standardize", {})
        self._standardize_enabled = standardize.get("enabled", False)
        self._standardize_cross_section = standardize.get("cross_section", False)
        self._standardize_expanding = standardize.get("expanding", False)
        self._standardize_window = standardize.get("window", None)
        self._standardize_auto_apply = standardize.get("auto_apply", False)
        
        # Clip config
        clip = preprocess.get("clip", {})
        self._clip_enabled = clip.get("enabled", False)
        self._clip_expanding = clip.get("expanding", False)
        self._clip_window = clip.get("window", None)
        self._clip_multiplier = clip.get("multiplier", None)
        self._clip_auto_apply = clip.get("auto_apply", False)
        
        # Winsorize config
        winsorize = preprocess.get("winsorize", {})
        self._winsorize_enabled = winsorize.get("enabled", False)
        self._winsorize_limits = winsorize.get("limits", None)
        self._winsorize_expanding = winsorize.get("expanding", False)
        self._winsorize_window = winsorize.get("window", None)
        self._winsorize_auto_apply = winsorize.get("auto_apply", False)
        
        # Industry neutralize config
        industry_neutralize = preprocess.get("industry_neutralize", {})
        self._industry_neutralize_enabled = industry_neutralize.get("enabled", False)
        self._industry_neutralize_method = industry_neutralize.get("method", None)
        self._industry_neutralize_classification = industry_neutralize.get("classification", None)
        self._industry_neutralize_auto_apply = industry_neutralize.get("auto_apply", False)

        #######################################################################
        
        # Mark initialization as complete
        self._initialized = True
        
        # Calculate dependent parameters immediately
        self._calc_extra_params()

    def _load_config(self):
        """Load and merge configurations from default and strategy-specific files"""
        # Load default system configuration
        default_cfg_path = f"{self.proj_path}/basis/config/default_config.yaml"
        default_cfg = yamler(default_cfg_path).get_all_fields()
        
        # If no calling file provided, use defaults only
        if not self.strategy_path:
            return default_cfg
            
        # Load strategy-specific configuration
        strategy_cfg_path = os.path.join(self.strategy_folder, "config.yaml")
        
        if os.path.exists(strategy_cfg_path):
            strategy_cfg = yamler(strategy_cfg_path).get_all_fields()
            # Merge strategy config over default config
            return collections.ChainMap(strategy_cfg, default_cfg)
        else:
            print(f"Warning: Strategy config not found at {strategy_cfg_path}, using defaults only")
            return default_cfg
    
    def validate(self, params_dict=None):
        """Validate the current configuration or a specified dictionary"""
        packed_dict = self.all_config.copy() if not params_dict else params_dict
        # exclude factor config
        exclude_key = "factor_config"
        packed_dict.pop(exclude_key, None)
        self.flattened_dict = unpack_dict_for_init(packed_dict)
        return validate_parameters(self.flattened_dict)
    
    def _calc_extra_params(self):
        """Calculate dependent parameters based on current config"""
        # calculate extra dates to ensure valid value from start_date
        self.calendar = Calendar(cal_type_or_config=CalendarTypes(self.region))
        self.start_date_extend = (self.calendar.adjust(
            self.start_date, -self.cov_window
            ).strftime("%Y%m%d") if self.cov_window else self.start_date)
        self.start_date_lookback = self.calendar.adjust(
            self.start_date, -self.lookback_window
            ).strftime("%Y%m%d") if self.lookback_window else self.start_date
    
    # Basic Config Properties
    @tracked_property("basic_config", "database")
    def database(self): pass

    # Strategy Config Properties
    @tracked_property("strategy_config", "benchmark")
    def benchmark(self): pass
    
    @tracked_property("strategy_config", "index_code")
    def index_code(self): pass
    
    @tracked_property("strategy_config", "start_date", triggers_calc=True)
    def start_date(self): pass
    
    @tracked_property("strategy_config", "end_date")
    def end_date(self): pass
    
    @tracked_property("strategy_config", "asset_range")
    def asset_range(self): pass
    
    @tracked_property("strategy_config", "date_rule")
    def date_rule(self): pass
    
    # Run Config Properties
    @tracked_property("run_config", "engine")
    def engine(self): pass
    
    @tracked_property("run_config", "region", triggers_calc=True)
    def region(self): pass
    
    @tracked_property("run_config", "trade_unit")
    def trade_unit(self): pass
    
    @tracked_property("run_config", "limit_threshold")
    def limit_threshold(self): pass
    
    @tracked_property("run_config", "sell_rule")
    def sell_rule(self): pass
    
    @tracked_property("run_config", "seed")
    def seed(self): pass
    
    @tracked_property("run_config", "enable_gif")
    def enable_gif(self): pass
    
    # Client Config Properties
    @tracked_property("client_config", "init_capital")
    def init_capital(self): pass
    
    @tracked_property("client_config.slippage", "slippage_type")
    def slippage_type(self): pass
    
    @tracked_property("client_config.slippage", "slippage_rate")
    def slippage_rate(self): pass
    
    @tracked_property("client_config", "commission")
    def commission(self): pass
    
    @tracked_property("client_config", "stamp_duty")
    def stamp_duty(self): pass
    
    @tracked_property("client_config", "min_cost")
    def min_cost(self): pass
    
    # Portfolio Config Properties
    @tracked_property("portfolio_config", "optimizer")
    def optimizer(self): pass
    
    @tracked_property("portfolio_config.cov_est", "cov_method")
    def cov_method(self): pass
    
    @tracked_property("portfolio_config.cov_est", "cov_window", triggers_calc=True)
    def cov_window(self): pass
    
    @tracked_property("portfolio_config", "opt_func")
    def opt_func(self): pass
    
    @tracked_property("portfolio_config", "long_short")
    def long_short(self): pass
    
    # Risk Config Properties
    @tracked_property("risk_config", "take_profit")
    def take_profit(self): pass
    
    @tracked_property("risk_config", "stop_loss")
    def stop_loss(self): pass

    @tracked_property("risk_config", "max_position_size")
    def max_position_size(self): pass
    
    @tracked_property("risk_config", "sector_concentration")
    def sector_concentration(self): pass
    
    @tracked_property("risk_config", "rebalance_threshold")
    def rebalance_threshold(self): pass
    
    # Factor Config Properties
    @tracked_property("factor_config.calc", "lookback_window", triggers_calc=True)
    def lookback_window(self): pass

    @tracked_property("factor_config.filter.universe", "include_SH")
    def include_SH(self): pass
    
    @tracked_property("factor_config.filter.universe", "include_SZ")
    def include_SZ(self): pass
    
    @tracked_property("factor_config.filter.universe", "include_BJ")
    def include_BJ(self): pass
    
    @tracked_property("factor_config.filter.universe", "include_GEM")
    def include_GEM(self): pass
    
    @tracked_property("factor_config.filter.universe", "include_STAR")
    def include_STAR(self): pass
    
    # Filter Universe Name Properties
    @tracked_property("factor_config.filter.universe", "include_ST")
    def include_ST(self): pass
    
    @tracked_property("factor_config.filter.universe", "include_delist")
    def include_delist(self): pass

    @tracked_property("factor_config.filter.universe", "min_list_days")
    def min_list_days(self): pass

    # Factor filter condition
    @tracked_property("factor_config.filter", "condition") 
    def filter_condition(self): pass

    # Factor filter target date
    @tracked_property("factor_config.filter", "target_date")
    def filter_target_date(self): pass

    # Factor Preprocess Properties
    # Pipeline config
    @tracked_property("factor_config.preprocess.pipeline", "order")
    def preprocess_order(self): pass
    
    @tracked_property("factor_config.preprocess.pipeline", "auto_apply")
    def preprocess_auto_apply(self): pass
    
    # Normalize config
    @tracked_property("factor_config.preprocess.normalize", "enabled")
    def normalize_enabled(self): pass
    
    @tracked_property("factor_config.preprocess.normalize", "cross_section")
    def normalize_cross_section(self): pass
    
    @tracked_property("factor_config.preprocess.normalize", "expanding")
    def normalize_expanding(self): pass
    
    @tracked_property("factor_config.preprocess.normalize", "window")
    def normalize_window(self): pass
    
    @tracked_property("factor_config.preprocess.normalize", "auto_apply")
    def normalize_auto_apply(self): pass
    
    # Standardize config
    @tracked_property("factor_config.preprocess.standardize", "enabled")
    def standardize_enabled(self): pass
    
    @tracked_property("factor_config.preprocess.standardize", "cross_section")
    def standardize_cross_section(self): pass
    
    @tracked_property("factor_config.preprocess.standardize", "expanding")
    def standardize_expanding(self): pass
    
    @tracked_property("factor_config.preprocess.standardize", "window")
    def standardize_window(self): pass
    
    @tracked_property("factor_config.preprocess.standardize", "auto_apply")
    def standardize_auto_apply(self): pass
    
    # Clip config
    @tracked_property("factor_config.preprocess.clip", "enabled")
    def clip_enabled(self): pass
    
    @tracked_property("factor_config.preprocess.clip", "expanding")
    def clip_expanding(self): pass
    
    @tracked_property("factor_config.preprocess.clip", "window")
    def clip_window(self): pass
    
    @tracked_property("factor_config.preprocess.clip", "multiplier")
    def clip_multiplier(self): pass
    
    @tracked_property("factor_config.preprocess.clip", "auto_apply")
    def clip_auto_apply(self): pass
    
    # Winsorize config
    @tracked_property("factor_config.preprocess.winsorize", "enabled")
    def winsorize_enabled(self): pass
    
    @tracked_property("factor_config.preprocess.winsorize", "limits")
    def winsorize_limits(self): pass
    
    @tracked_property("factor_config.preprocess.winsorize", "expanding")
    def winsorize_expanding(self): pass
    
    @tracked_property("factor_config.preprocess.winsorize", "window")
    def winsorize_window(self): pass
    
    @tracked_property("factor_config.preprocess.winsorize", "auto_apply")
    def winsorize_auto_apply(self): pass
    
    # Industry neutralize config
    @tracked_property("factor_config.preprocess.industry_neutralize", "enabled")
    def industry_neutralize_enabled(self): pass
    
    @tracked_property("factor_config.preprocess.industry_neutralize", "method")
    def industry_neutralize_method(self): pass
    
    @tracked_property("factor_config.preprocess.industry_neutralize", "classification")
    def industry_neutralize_classification(self): pass
    
    @tracked_property("factor_config.preprocess.industry_neutralize", "auto_apply")
    def industry_neutralize_auto_apply(self): pass

    def reload_config(self, calling_file):
        """Reload configuration with a different strategy file"""
        self.__init__(calling_file)
    
    def get_strategy_specific_config(self, section):
        """Get strategy-specific configuration section"""
        return self.all_config.get(section, {})
    
    def update_config(self, path_or_attr, value):
        """
        Update a configuration value through property system or nested path
        
        Args:
            path_or_attr (str): Either an attribute name or dot-separated path
            value: The value to set
            
        Examples:
            config.update_config('start_date', '20250616')  # Uses property
            config.update_config('portfolio_config.cov_est.cov_method', 'custom')  # Nested path
        """
        # First try as a property attribute
        if hasattr(self, path_or_attr) and hasattr(getattr(self.__class__, path_or_attr, None), "__set__"):
            # Use property setter to ensure all logic is triggered
            setattr(self, path_or_attr, value)
        elif "." in path_or_attr:
            # Handle as nested path
            self._set_nested_config(path_or_attr, value)
        else:
            raise AttributeError(f"'{path_or_attr}' is not a tracked configuration attribute or valid nested path")
    
    def get_config(self, path_or_attr, default=None):
        """
        Get configuration values through property system or nested path
        
        Args:
            path_or_attr (str): Either an attribute name or dot-separated path
            default: Default value if path doesn't exist
            
        Examples:
            config.get_config('start_date')  # Gets property value
            config.get_config('portfolio_config.cov_est.cov_method')  # Nested path
            
        Returns:
            The value at the specified path/attribute or default
        """
        # First try as a property attribute
        if hasattr(self, path_or_attr) and hasattr(getattr(self.__class__, path_or_attr, None), "__get__"):
            # Use property getter
            return getattr(self, path_or_attr)
        elif "." in path_or_attr:
            # Handle as nested path
            keys = path_or_attr.split(".")
            config = self.all_config
            
            try:
                for key in keys:
                    config = config[key]
                return config
            except (KeyError, TypeError):
                return default
        else:
            # Try as simple attribute (non-property)
            return getattr(self, path_or_attr, default)
    
    def get_all_params(self):
        # return all tracked_property defined config attributes
        params_dict = {}
        for attr_name in dir(self.__class__):
            attr_obj = getattr(self.__class__, attr_name)
            if isinstance(attr_obj, property) and not attr_name.startswith('_'):
                params_dict[attr_name] = getattr(self, attr_name)
        
        # add other public attributes
        for k, v in self.__dict__.items():
            if not k.startswith("_") and k not in params_dict:
                params_dict[k] = v
        
        return params_dict

# Factory function for creating strategy-specific configs
def get_strategy_config(calling_file):
    """Create a FrozenConfig instance for a specific strategy"""
    return FrozenConfig(calling_file=calling_file)
