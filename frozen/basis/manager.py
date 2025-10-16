import os
import yaml
from typing import Dict, Any, Optional
from .constant import get_strategy_config


class ConfigManager:
    """Configuration manager for strategy-level configurations"""
    
    @staticmethod
    def create_strategy_config_template(strategy_path: str, strategy_name: str) -> str:
        """
        Create a configuration template for a new strategy
        
        Args:
            strategy_path (str): Path to strategy folder
            strategy_name (str): Name of the strategy
            
        Returns:
            str: Path to created config file
        """
        config_template = {
            'strategy_meta': {
                'name': strategy_name,
                'description': f"{strategy_name} trading strategy",
                'version': "1.0.0",
                'author': "Strategy Team"
            },
            'base_config': {
                'database': 'duckdb'
            },
            'run_config': {
                'region': 'cn',
                'trade_unit': 100,
                'limit_threshold': 0.1,
                'sell_rule': 'currentBar',
                'seed': 42,
                'enable_gif': False
            },
            'client_config': {
                'init_capital': 1000000,
                'slippage': {
                    'slippage_type': 'byRate',
                    'slippage_rate': 0.0001
                },
                'commission': 0.0003,
                'stamp_duty': 0.001,
                'min_cost': 5
            },
            'strategy_config': {
                'benchmark': '000300.SH',
                'index_code': '000300.SH',
                'start_date': '20200101',
                'end_date': '20240702',
                'asset_range': "(0, 10)",
                'date_rule': 'W-WED'
            },
            'portfolio_config': {
                'optimizer': 'mean-variance',
                'cov_est': {
                    'cov_method': 'custom',
                    'cov_window': 60
                },
                'opt_func': 'sharpe',
                'long_short': False
            },
            'risk_config': {
                'take_profit': float('inf'),
                'stop_loss': float('-inf'),
                'max_position_size': 0.1,
                'sector_concentration': 0.3,
                'rebalance_threshold': 0.05
            },
            'factor_config': {
                'calc': {
                    'lookback_window': 60
                },
                'filter': {
                    'universe': {
                        'include_SH': True,
                        'include_SZ': True,
                        'include_BJ': False,
                        'include_GEM': False,
                        'include_STAR': False,
                        'include_ST': False,
                        'include_delist': False,
                        'min_list_days': 100
                    }
                }
            }
        }
        
        # Ensure directory exists
        os.makedirs(strategy_path, exist_ok=True)
        
        config_path = os.path.join(strategy_path, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_template, f, default_flow_style=False, allow_unicode=True)
        
        return config_path
    
    @staticmethod
    def validate_strategy_config(strategy_path_or_file: str) -> Dict[str, Any]:
        """
        Validate strategy configuration
        
        Args:
            strategy_path_or_file (str): Path to strategy folder OR strategy file
            - If folder: will look for config.yaml in that folder
            - If .py file: will use that file directly
            
        Returns:
            Dict containing validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Determine if input is a file or folder
        if os.path.isfile(strategy_path_or_file) and strategy_path_or_file.endswith('.py'):
            # Real strategy file provided
            strategy_file = strategy_path_or_file
            strategy_folder = os.path.dirname(strategy_file)
        elif os.path.isdir(strategy_path_or_file):
            # Strategy folder provided, create dummy file path
            strategy_folder = strategy_path_or_file
            strategy_file = os.path.join(strategy_folder, "strategy.py")
        else:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid path: {strategy_path_or_file}. Must be a strategy folder or .py file")
            return validation_result
        
        # Check if config.yaml exists
        config_path = os.path.join(strategy_folder, 'config.yaml')
        if not os.path.exists(config_path):
            validation_result['valid'] = False
            validation_result['errors'].append(f"Config file not found at {config_path}")
            return validation_result
        
        try:
            # Load configuration using FrozenConfig (automatically validates during __init__)
            config = get_strategy_config(strategy_file)
            
            # If we reach here, the built-in validation passed
            validation_result['warnings'].append("Configuration passed built-in validation")
            
            # Additional business logic validations
            # Check required sections
            required_sections = ['strategy_config']
            for section in required_sections:
                section_config = config.get_strategy_specific_config(section)
                if not section_config:
                    validation_result['warnings'].append(f"Section '{section}' not found, using defaults")
            
            # Additional business rule validations
            try:
                # Date range validation
                start_date = config.start_date
                end_date = config.end_date
                if len(start_date) != 8 or len(end_date) != 8:
                    validation_result['errors'].append("Date format should be YYYYMMDD")
                    validation_result['valid'] = False
                if start_date >= end_date:
                    validation_result['errors'].append("start_date must be earlier than end_date")
                    validation_result['valid'] = False
            except Exception as e:
                validation_result['errors'].append(f"Invalid date configuration: {e}")
                validation_result['valid'] = False
            
            
            try:
                # Validate asset_range
                asset_range = config.asset_range
                if not isinstance(asset_range, (list, tuple)) or len(asset_range) != 2:
                    validation_result['errors'].append("asset_range should be a tuple/list of two elements")
                    validation_result['valid'] = False
            except Exception as e:
                validation_result['errors'].append(f"Invalid asset_range configuration: {e}")
                validation_result['valid'] = False
                
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Failed to load configuration: {e}")
        
        return validation_result
    
    @staticmethod
    def list_available_strategies(base_path: str = "demo/strategy") -> Dict[str, Dict[str, Any]]:
        """
        List all available strategies with their metadata
        
        Args:
            base_path (str): Base path to search for strategies
            
        Returns:
            Dict containing strategy information
        """
        strategies = {}
        
        if not os.path.exists(base_path):
            return strategies
            
        for item in os.listdir(base_path):
            strategy_path = os.path.join(base_path, item)
            
            if os.path.isdir(strategy_path):
                config_path = os.path.join(strategy_path, 'config.yaml')
                
                if os.path.exists(config_path):
                    try:
                        # Create a dummy strategy file path for loading config
                        dummy_strategy_file = os.path.join(strategy_path, "strategy.py")
                        config = get_strategy_config(dummy_strategy_file)
                        
                        # Get strategy metadata from config
                        strategy_meta = config.get_strategy_specific_config('strategy_meta')
                        
                        strategies[item] = {
                            'name': strategy_meta.get('name', item),
                            'description': strategy_meta.get('description', f"{item} trading strategy"),
                            'version': strategy_meta.get('version', '1.0.0'),
                            'author': strategy_meta.get('author', 'Unknown'),
                            'path': strategy_path,
                            'valid': ConfigManager.validate_strategy_config(strategy_path)['valid']
                        }
                    except Exception as e:
                        strategies[item] = {
                            'name': item,
                            'description': 'Failed to load configuration',
                            'version': 'unknown',
                            'author': 'unknown',
                            'path': strategy_path,
                            'valid': False,
                            'error': str(e)
                        }
        
        return strategies
    
    @staticmethod
    def copy_strategy_template(source_strategy: str, target_strategy: str, base_path: str = "demo/strategy") -> str:
        """
        Copy configuration from one strategy to create a new one
        
        Args:
            source_strategy (str): Source strategy name
            target_strategy (str): Target strategy name
            base_path (str): Base path for strategies
            
        Returns:
            str: Path to new strategy config
        """
        import shutil
        
        source_path = os.path.join(base_path, source_strategy)
        target_path = os.path.join(base_path, target_strategy)
        
        if not os.path.exists(source_path):
            raise ValueError(f"Source strategy not found: {source_path}")
        
        if os.path.exists(target_path):
            raise ValueError(f"Target strategy already exists: {target_path}")
        
        # Copy the entire strategy folder
        shutil.copytree(source_path, target_path)
        
        # Update the config file with new strategy name
        config_path = os.path.join(target_path, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'strategy_meta' in config:
                config['strategy_meta']['name'] = target_strategy
                config['strategy_meta']['description'] = f"{target_strategy} trading strategy (copied from {source_strategy})"
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return config_path 