"""
Command-line interface for managing strategy configurations
"""
import argparse
import os

from frozen.basis.manager import ConfigManager
from frozen.basis.constant import get_strategy_config


def create_strategy(args):
    """Create a new strategy configuration"""
    strategy_path = os.path.join(args.base_path, args.name)
    
    try:
        config_path = ConfigManager.create_strategy_config_template(strategy_path, args.name)
        print(f"✅ Created strategy configuration at: {config_path}")
        
        # Validate the created config
        validation = ConfigManager.validate_strategy_config(strategy_path)
        if validation['valid']:
            print(f"✅ Configuration is valid")
        else:
            print(f"⚠️ Configuration has issues:")
            for error in validation['errors']:
                print(f"   ❌ {error}")
    except Exception as e:
        print(f"❌ Failed to create strategy: {e}")


def list_strategies(args):
    """List all available strategies"""
    strategies = ConfigManager.list_available_strategies(args.base_path)
    
    if not strategies:
        print(f"No strategies found in {args.base_path}")
        return
    
    print(f"\n📋 Available Strategies in {args.base_path}:")
    print("=" * 60)
    
    for folder_name, info in strategies.items():
        status = "✅" if info['valid'] else "❌"
        print(f"{status} {info['name']} (v{info['version']})")
        print(f"   📁 {info['path']}")
        print(f"   📝 {info['description']}")
        print(f"   👤 {info['author']}")
        if not info['valid'] and 'error' in info:
            print(f"   ⚠️ Error: {info['error']}")
        print()


def validate_strategy(args):
    """Validate a strategy configuration"""
    strategy_path = os.path.join(args.base_path, args.name)
    
    validation = ConfigManager.validate_strategy_config(strategy_path)
    
    print(f"\n🔍 Validating strategy: {args.name}")
    print("=" * 40)
    
    if validation['valid']:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"   ❌ {error}")
    
    if validation['warnings']:
        print("\n⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"   ⚠️ {warning}")


def show_config(args):
    """Show strategy configuration details"""
    strategy_path = os.path.join(args.base_path, args.name)
    
    try:
        # Create a dummy strategy file path for loading config
        dummy_strategy_file = os.path.join(strategy_path, "strategy.py")
        config = get_strategy_config(dummy_strategy_file)
        
        # Get strategy metadata
        strategy_meta = config.get_strategy_specific_config('strategy_meta')
        
        print(f"\n📊 Configuration for: {args.name}")
        print("=" * 50)
        
        print(f"📈 Strategy: {strategy_meta.get('name', args.name)} (v{strategy_meta.get('version', '1.0.0')})")
        print(f"📝 Description: {strategy_meta.get('description', 'No description')}")
        print(f"👤 Author: {strategy_meta.get('author', 'Unknown')}")
        print(f"📅 Period: {config.start_date} to {config.end_date}")
        print(f"🎯 Benchmark: {config.benchmark}")
        print(f"📊 Asset Range: {config.asset_range}")
        print(f"🔄 Rebalance: {config.date_rule}")
        print(f"⚙️ Optimizer: {config.optimizer}")
        print(f"💰 Initial Capital: {config.init_capital:,}")
        print(f"🛡️ Max Position: {config.max_position_size:.1%}")
        print(f"🏢 Sector Concentration: {config.sector_concentration:.1%}")
        print(f"📈 Take Profit: {config.take_profit}")
        print(f"📉 Stop Loss: {config.stop_loss}")
        print(f"🌍 Region: {config.region}")
        print(f"🎲 Seed: {config.seed}")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")


def copy_strategy(args):
    """Copy strategy configuration from existing strategy"""
    try:
        config_path = ConfigManager.copy_strategy_template(
            args.source, args.target, args.base_path
        )
        print(f"✅ Copied strategy from '{args.source}' to '{args.target}'")
        print(f"📄 Configuration: {config_path}")
        
        # Validate the copied config
        strategy_path = os.path.join(args.base_path, args.target)
        validation = ConfigManager.validate_strategy_config(strategy_path)
        if validation['valid']:
            print(f"✅ Configuration is valid")
        else:
            print(f"⚠️ Configuration has issues:")
            for error in validation['errors']:
                print(f"   ❌ {error}")
                
    except Exception as e:
        print(f"❌ Failed to copy strategy: {e}")


def main():
    parser = argparse.ArgumentParser(description="Strategy Configuration Manager")
    parser.add_argument("--base-path", default="demo/strategy", 
                       help="Base path for strategies (default: demo/strategy)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create strategy command
    create_parser = subparsers.add_parser("create", help="Create new strategy")
    create_parser.add_argument("name", help="Strategy name")
    create_parser.set_defaults(func=create_strategy)
    
    # List strategies command
    list_parser = subparsers.add_parser("list", help="List all strategies")
    list_parser.set_defaults(func=list_strategies)
    
    # Validate strategy command
    validate_parser = subparsers.add_parser("validate", help="Validate strategy config")
    validate_parser.add_argument("name", help="Strategy name")
    validate_parser.set_defaults(func=validate_strategy)
    
    # Show config command
    show_parser = subparsers.add_parser("show", help="Show strategy configuration")
    show_parser.add_argument("name", help="Strategy name")
    show_parser.set_defaults(func=show_config)
    
    # Copy strategy command
    copy_parser = subparsers.add_parser("copy", help="Copy strategy from existing one")
    copy_parser.add_argument("source", help="Source strategy name")
    copy_parser.add_argument("target", help="Target strategy name")
    copy_parser.set_defaults(func=copy_strategy)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 