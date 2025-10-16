import os
import sys
import logging
import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..basis import FrozenConfig


class MakeFile:
    """Create relevant files for strategy."""

    def __init__(self, config: "FrozenConfig"):

        self.strategy_folder = config.strategy_folder
        self.strategy_name = config.strategy_name
        self.FULL_PATH = os.path.join(self.strategy_folder, "performance_report")
    
    def create_folder(self):
        if not os.path.exists(self.FULL_PATH):
            os.makedirs(self.FULL_PATH)

    def filename(self, calling_file):

        script_path = os.path.abspath(calling_file)
        script_filename = os.path.basename(script_path)

        return script_filename


class FrozenLogger(MakeFile):
    """Logger for Frozen Framework"""
    
    def __init__(self, config: "FrozenConfig", run_mode: str):
        super().__init__(config)
        self.create_folder()
        self.logger = self._init_logger(run_mode)
    
    def get_logger(self):
        """Get the configured strategy logger"""
        return self.logger
    
    def _init_logger(self, run_mode: str):
        """
        Generate the strategy trading log.
        
        Parameters:
        -----------
        run_mode: str
            The operating mode of the backtest engine.
            - "normal": The default processing mode.
            - "parallel": The multi-processing or multi-threading mode.
        """

        LOG_NAME = "account activities.log"
        LOGGER_NAME = f"frozen.{self.strategy_name}"

        logger = logging.getLogger(LOGGER_NAME)
        
        if run_mode == "normal":
            logger.setLevel(logging.INFO)
            
            file_handler = logging.FileHandler(filename=os.path.join(self.FULL_PATH, LOG_NAME), mode="w")
            file_handler.setLevel(logging.INFO)

            # create two different formatters
            header_formatter = logging.Formatter("%(message)s")  # simple formatter, only display message
            main_formatter = logging.Formatter(
                fmt="[%(levelname)s] %(name)s [%(filename)s:%(lineno)d] [%(trade_date)s]: %(message)s", 
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # use header formatter to record the start time of the backtest
            file_handler.setFormatter(header_formatter)
            logger.addHandler(file_handler)
            
            # switch to main formatter for subsequent logs
            file_handler.setFormatter(main_formatter)

        else:  # parallel mode
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.addHandler(logging.NullHandler())

        logger.propagate = False

        class TradeLogger(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                if "extra" not in kwargs:
                    kwargs["extra"] = {}
                if "trade_date" not in kwargs["extra"]:
                    kwargs["extra"]["trade_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
                return msg, kwargs

        logger = TradeLogger(logger, {})

        # record the start time of the backtest
        current_time = datetime.datetime.now()
        logger.info(f"Strategy backtest started at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("-" * 40)
        
        # # log the start time of the backtest
        # current_time = datetime.datetime.now()
        # logger.info(
        #     f"Strategy backtest started at {current_time.strftime('%Y-%m-%d %H:%M:%S')}",
        #     extra={'trade_date': current_time.strftime('%Y-%m-%d')}
        # )
        # logger.info("-" * 80, extra={})  # add separator line

        return logger


class GeneralLogger:
    """General logger, print to terminal only"""
    
    def __init__(self, 
                 default_level: int = logging.INFO,
                 format_str: Optional[str] = None,
                 muted: bool = False):
        """
        Initialize the general logger
        
        :param default_level: Default logging level
        :param format_str: Custom log format string
        :param muted: Whether to mute all logging output
        """
        self.default_level = default_level
        self.format_str = format_str or (
            "[%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
        )
        self.muted = muted
        self._configure_root_logger()
    
    def reset_all_loggers(self):
        """Reset all existing loggers to clean state and reconfigure according to current GL state"""
        for name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
                existing_logger = logging.getLogger(name)
                
                # Remove all handlers
                for handler in existing_logger.handlers[:]:
                    existing_logger.removeHandler(handler)
                
                # Reset level to NOTSET (inherit from parent)
                existing_logger.setLevel(logging.NOTSET)
                
                # Enable propagation
                existing_logger.propagate = True
                
                # Restore original handle method if it was overridden
                if hasattr(existing_logger, '_original_handle'):
                    existing_logger.handle = existing_logger._original_handle
                    delattr(existing_logger, '_original_handle')
        
        # Reconfigure root logger
        self._configure_root_logger()
        
        # Apply current muting state to all loggers
        self._update_existing_loggers()
    
    def _configure_root_logger(self):
        """Configure root logger"""
        root_logger = logging.getLogger()
        
        # Remove all existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        if self.muted:
            # Add null handler to mute all output
            root_logger.addHandler(logging.NullHandler())
            root_logger.setLevel(logging.CRITICAL + 1)  # Set to a level higher than CRITICAL
        else:
            root_logger.setLevel(self.default_level)
            # Create and add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(self.format_str)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        Get a logger with the specified name
        
        :param name: Logger name
        :param level: Optional, logging level
        :return: Configured logger instance
        """
        logger = logging.getLogger(name)
        
        if self.muted:
            # If muted, disable this logger completely and isolate it
            logger.setLevel(logging.CRITICAL + 1)
            logger.propagate = False
            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Add null handler
            logger.addHandler(logging.NullHandler())
            
            # For extra protection, also disable the root logger handlers for this specific logger
            # by overriding the handle method
            if not hasattr(logger, '_original_handle'):
                logger._original_handle = logger.handle
            def silent_handle(_record):
                # Completely discard all log records
                return
            logger.handle = silent_handle
            
        else:
            # Set logging level
            if level is not None:
                logger.setLevel(level)
            else:
                logger.setLevel(self.default_level)
            
            # Enable propagation
            logger.propagate = True
            
            # Restore original handle method if it was overridden
            if hasattr(logger, '_original_handle'):
                logger.handle = logger._original_handle
                delattr(logger, '_original_handle')
        
        return logger

    def mute(self):
        """Mute all logging output"""
        self.muted = True
        self._configure_root_logger()
        self._update_existing_loggers()
    
    def unmute(self):
        """Unmute logging output"""
        self.muted = False
        
        # Ensure clean state before reconfiguration
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Clean up all existing logger handlers
        for name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
        
        self._configure_root_logger()
        self._update_existing_loggers()
    
    def _update_existing_loggers(self):
        """Update all existing loggers to match current muted state"""
        # Get all existing loggers
        for name in logging.Logger.manager.loggerDict:
            if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
                existing_logger = logging.getLogger(name)
                
                if self.muted:
                    # Mute existing logger
                    existing_logger.setLevel(logging.CRITICAL + 1)
                    existing_logger.propagate = False
                    # Remove any existing handlers
                    for handler in existing_logger.handlers[:]:
                        existing_logger.removeHandler(handler)
                    # Add null handler
                    existing_logger.addHandler(logging.NullHandler())
                    # Override handle method if not already done
                    if not hasattr(existing_logger, '_original_handle'):
                        existing_logger._original_handle = existing_logger.handle
                    def silent_handle(_record):
                        return
                    existing_logger.handle = silent_handle
                else:
                    # Restore existing logger
                    existing_logger.setLevel(self.default_level)
                    existing_logger.propagate = True
                    # Remove null handlers
                    for handler in existing_logger.handlers[:]:
                        if isinstance(handler, logging.NullHandler):
                            existing_logger.removeHandler(handler)
                    # Restore original handle method if it was overridden
                    if hasattr(existing_logger, '_original_handle'):
                        existing_logger.handle = existing_logger._original_handle
                        delattr(existing_logger, '_original_handle')
    
    def is_muted(self) -> bool:
        """Check if logger is muted"""
        return self.muted
    
    @staticmethod
    def set_log_level(logger: logging.Logger, level: int):
        """Set logging level for a specific logger"""
        logger.setLevel(level)


GL = GeneralLogger(muted=True)  # Default to muted state
GL.reset_all_loggers()  # Ensure clean state on module import