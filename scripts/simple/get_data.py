import fire
import argparse
import sys
import signal
from frozen.workflow.data import GetData
from frozen.data.provider import ProviderTypes
from frozen.data.database import DatabaseTypes
from frozen.utils.calendar import CalendarTypes

def signal_handler(sig, frame):
    """Handle keyboard interrupt"""
    print('\nReceived interrupt signal, exiting...')
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data from various sources')
    parser.add_argument('--provider', type=str, default='TUSHARE', help='Data source provider (e.g., TUSHARE)')
    parser.add_argument('--database', type=str, default='DUCKDB', help='Database type (e.g., MONGODB, DUCKDB, CHDB)')
    parser.add_argument('--calendar', type=str, default='CHINA', help='Market calendar type (e.g., CHINA)')
    parser.add_argument('--update', action='store_true', help='Whether to update existing data')
    parser.add_argument('--start_date', type=str, help='Start date in format YYYYMMDD')
    parser.add_argument('--end_date', type=str, help='End date in format YYYYMMDD')
    parser.add_argument('--parallel', action='store_true', help='Whether to use multi-threading')
    parser.add_argument('command', nargs='?', default='frozen_data', help='Command to execute (e.g., frozen_data)')
    args, unknown_args = parser.parse_known_args()
    
    provider = None
    database = None
    calendar = None
    
    if args.provider:
        provider_str = args.provider.upper()
        try:
            provider = getattr(ProviderTypes, provider_str)
        except AttributeError:
            print(f"Wrong provider type: '{args.provider}'")
            print(f"Available provider types: {', '.join([t.name for t in ProviderTypes])}")
            sys.exit(1)
    
    if args.database:
        database_str = args.database.upper()
        try:
            database = getattr(DatabaseTypes, database_str)
        except AttributeError:
            print(f"Wrong database type: '{args.database}'")
            print(f"Available database types: {', '.join([t.name for t in DatabaseTypes])}")
            sys.exit(1)
    
    if args.calendar:
        calendar_str = args.calendar.upper()
        try:
            calendar = getattr(CalendarTypes, calendar_str)
        except AttributeError:
            print(f"Wrong calendar type: '{args.calendar}'")
            print(f"Available calendar types: {', '.join([t.name for t in CalendarTypes])}")
            sys.exit(1)
    
    # Parameters for creating GetData instance
    init_kwargs = {}
    if provider is not None:
        init_kwargs['provider_type'] = provider
    if database is not None:
        init_kwargs['database_type'] = database
    if calendar is not None:
        init_kwargs['calendar_type'] = calendar
    
    # Parameters for method calls
    method_kwargs = {}
    if args.start_date:
        method_kwargs['start_date'] = args.start_date
    if args.end_date:
        method_kwargs['end_date'] = args.end_date
    method_kwargs['update'] = args.update
    method_kwargs['parallel'] = args.parallel
    
    # Create GetData instance and call the corresponding method
    data_getter = GetData(**init_kwargs)
    
    try:
        if args.command:
            if hasattr(data_getter, args.command):
                method = getattr(data_getter, args.command)
                method(**method_kwargs)
            else:
                print(f"Unknown command: {args.command}")
                print(f"Available commands: {', '.join([m for m in dir(data_getter) if not m.startswith('_') and callable(getattr(data_getter, m))])}")
                sys.exit(1)
        else:
            # If no command is specified, execute frozen_data by default
            if hasattr(data_getter, "frozen_data"):
                data_getter.frozen_data(**method_kwargs)
            else:
                print("Default command 'frozen_data' not found")
                sys.exit(1)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt detected, program exiting...')
        sys.exit(0)
    except Exception as e:
        print(f'Program execution error: {e}')
        sys.exit(1)