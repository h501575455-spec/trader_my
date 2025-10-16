import os
import sys
import logging
import threading
import weakref
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional


class DataLogger:
    """
    Robust logging system that protects against configuration corruption 
    by other modules and provides reliable logging functionality.
    """
    _instance: Optional["DataLogger"] = None
    _lock = threading.RLock()
    
    def __init__(self, log_file_path: str = "datafeed.log"):
        if DataLogger._instance is not None:
            return
            
        self.log_file_path = log_file_path
        self._setup_complete = False
        
        # Store original handler configuration
        self._original_handlers: Dict[str, list] = {}
        self._original_levels: Dict[str, int] = {}
        self._original_settings: Dict[str, dict] = {}
        
        # Create our own handlers that we control
        self._console_handler: Optional[logging.Handler] = None
        self._file_handler: Optional[logging.Handler] = None
        
        # Track loggers we've configured
        self._configured_loggers: Dict[str, weakref.ReferenceType] = {}
        
        self._setup_logging()
        DataLogger._instance = self
    
    def _setup_logging(self):
        """Setup logging configuration with protection against corruption"""
        with self._lock:
            if self._setup_complete:
                return
                
            # Determine log path
            current_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            log_path = os.path.join(current_path, "logs")
            
            # Create log directory if it doesn't exist
            os.makedirs(log_path, exist_ok=True)
            
            # Create formatters
            simple_formatter = logging.Formatter(
                fmt="[%(levelname)s] - %(asctime)s - %(name)s - [%(filename)s:%(lineno)d]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            detailed_formatter = logging.Formatter(
                fmt="[%(process)s:%(threadName)s](%(asctime)s) %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            # Create console handler
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(logging.INFO)
            self._console_handler.setFormatter(simple_formatter)
            
            # Create file handler
            log_file = os.path.join(log_path, self.log_file_path)
            self._file_handler = RotatingFileHandler(
                filename=log_file,
                mode="a",
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3
            )
            self._file_handler.setLevel(logging.INFO)
            self._file_handler.setFormatter(detailed_formatter)
            
            self._setup_complete = True
    
    def _ensure_logger_configured(self, logger_name: str) -> logging.Logger:
        """Ensure logger is properly configured, fixing corruption if needed"""
        with self._lock:
            logger = logging.getLogger(logger_name)
            
            # Check if logger needs (re)configuration
            needs_config = (
                logger.level > logging.INFO or 
                not logger.handlers or
                any(isinstance(h, logging.NullHandler) for h in logger.handlers) or
                logger.disabled
            )
            
            if needs_config:
                # Save current state if not already saved
                if logger_name not in self._original_handlers:
                    self._original_handlers[logger_name] = logger.handlers[:]
                    self._original_levels[logger_name] = logger.level
                    self._original_settings[logger_name] = {
                        "disabled": logger.disabled,
                        "propagate": logger.propagate,
                        "filters": logger.filters[:],
                    }
                
                # Clear existing handlers
                logger.handlers.clear()
                logger.filters.clear()
                
                # Reconfigure logger
                logger.setLevel(logging.INFO)
                logger.disabled = False
                logger.propagate = False
                
                # Add our handlers
                if self._console_handler:
                    logger.addHandler(self._console_handler)
                if self._file_handler:
                    logger.addHandler(self._file_handler)
                
                # Store weak reference to track this logger
                self._configured_loggers[logger_name] = weakref.ref(logger)
            
            # Override logger methods with robust versions to handle future corruption
            self._wrap_logger_methods(logger)
            
            return logger
    
    def _get_caller_info(self, frame_offset: int = 2):
        """Get caller filename and line number"""
        import inspect
        frame = inspect.currentframe()
        # Go back frame_offset frames to get the actual caller
        for _ in range(frame_offset):
            if frame:
                frame = frame.f_back
        
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        return filename, lineno
    
    def _format_message(self, msg, *args):
        """Format message with args if provided"""
        if args:
            try:
                return msg % args
            except (TypeError, ValueError):
                return f"{msg} {args}"
        return str(msg)
    
    def _wrap_logger_methods(self, logger: logging.Logger):
        """Wrap logger methods with robust fallback functionality"""
        # Store original methods if not already wrapped
        if not hasattr(logger, "_original_info"):
            logger._original_info = logger.info
            logger._original_warning = logger.warning
            logger._original_error = logger.error
            logger._original_debug = logger.debug
            logger._original_critical = logger.critical
            
            # Create robust wrapper functions that always use direct logging
            def robust_info(msg, *args, **kwargs):
                formatted_msg = self._format_message(msg, *args)
                filename, lineno = self._get_caller_info(2)
                self._direct_log(logging.INFO, formatted_msg, logger.name, filename, lineno)
            
            def robust_warning(msg, *args, **kwargs):
                formatted_msg = self._format_message(msg, *args)
                filename, lineno = self._get_caller_info(2)
                self._direct_log(logging.WARNING, formatted_msg, logger.name, filename, lineno)
            
            def robust_error(msg, *args, **kwargs):
                formatted_msg = self._format_message(msg, *args)
                filename, lineno = self._get_caller_info(2)
                self._direct_log(logging.ERROR, formatted_msg, logger.name, filename, lineno)
                    
            def robust_debug(msg, *args, **kwargs):
                formatted_msg = self._format_message(msg, *args)
                filename, lineno = self._get_caller_info(2)
                self._direct_log(logging.DEBUG, formatted_msg, logger.name, filename, lineno)
            
            def robust_critical(msg, *args, **kwargs):
                formatted_msg = self._format_message(msg, *args)
                filename, lineno = self._get_caller_info(2)
                self._direct_log(logging.CRITICAL, formatted_msg, logger.name, filename, lineno)
            
            # Replace methods with robust versions
            logger.info = robust_info
            logger.warning = robust_warning
            logger.error = robust_error
            logger.debug = robust_debug
            logger.critical = robust_critical
    
    def get_logger(self, name: str = "frozen") -> logging.Logger:
        """Get a properly configured logger, fixing corruption if necessary"""
        # Always ensure configuration and wrapping on every call
        return self._ensure_logger_configured(name)
    
    def log_message(self, level: int, message: str, logger_name: str = "frozen", 
                   filename: str = "", lineno: int = 0):
        """
        Robust logging method that works even when standard logger methods fail.
        This bypasses potential corruption issues by directly using handlers.
        """
        try:
            # Try standard logging first
            logger = self._ensure_logger_configured(logger_name)
            if logger.isEnabledFor(level) and not logger.disabled:
                logger.log(level, message)
                return
        except Exception:
            pass
        
        # Fallback: Direct handler emission
        self._direct_log(level, message, logger_name, filename, lineno)
    
    def _direct_log(self, level: int, message: str, logger_name: str, 
                   filename: str = "", lineno: int = 0):
        """Directly emit log messages through handlers when logger is corrupted"""
        
        # Create log record manually
        record = logging.LogRecord(
            name=logger_name,
            level=level,
            pathname=filename,
            lineno=lineno,
            msg=message,
            args=(),
            exc_info=None,
            func=None,
            sinfo=None
        )
        
        # Emit through our handlers directly
        try:
            if self._console_handler:
                self._console_handler.emit(record)
        except Exception:
            pass
            
        try:
            if self._file_handler:
                self._file_handler.emit(record)
        except Exception:
            pass
    
    def info(self, message: str, logger_name: str = "frozen"):
        """Robust info logging"""
        import inspect
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        self.log_message(logging.INFO, message, logger_name, filename, lineno)
    
    def warning(self, message: str, logger_name: str = "frozen"):
        """Robust warning logging"""
        import inspect
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        self.log_message(logging.WARNING, message, logger_name, filename, lineno)
    
    def error(self, message: str, logger_name: str = "frozen"):
        """Robust error logging"""
        import inspect
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        self.log_message(logging.ERROR, message, logger_name, filename, lineno)
    
    def debug(self, message: str, logger_name: str = "frozen"):
        """Robust debug logging"""
        import inspect
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        self.log_message(logging.DEBUG, message, logger_name, filename, lineno)
    
    def critical(self, message: str, logger_name: str = "frozen"):
        """Robust critical logging"""
        import inspect
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename if frame else ""
        lineno = frame.f_lineno if frame else 0
        self.log_message(logging.CRITICAL, message, logger_name, filename, lineno)


# Create singleton instances 
L = DataLogger()

__all__ = ["L", "DataLogger"]
