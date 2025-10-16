"""
CSV to DuckDB Data Loader

A comprehensive data loading coordinator that combines CSV file reading
with DuckDB database operations for efficient data processing.

This module serves as the main interface, coordinating between:
- CSVFileReader: File discovery and reading
- DuckDBHandler: Database operations and storage
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Optional

from ...utils.log import L
from .filereader import CSVFileReader, FileInfo
from ...database.duck.handler import DuckDBHandler

logger = L.get_logger("frozen")


class CSVDataFeed:
    """
    Main coordinator class for loading CSV files into a DuckDB table.
    
    This class combines file reading and database operations in a clean,
    separated architecture for better maintainability and testability.
    
    Features:
    - Automatic file discovery and pattern matching
    - Schema detection and optimization
    - Batch processing with progress tracking
    - Incremental loading with duplicate detection
    - Parallel processing support
    - Comprehensive logging and error handling
    - Table storage: Store all CSV data in one table with metadata columns
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        table_name: str = "csv_data",
        batch_size: int = 10000,
        max_workers: int = 4,
        create_indexes: bool = True,
        engine: str = "pandas"
    ):
        """
        Initialize the CSV to DuckDB loader for table storage.
        
        Args:
            db_path: Path to DuckDB database file (optional)
            table_name: Name of table to store all data
            batch_size: Number of rows to process in each batch
            max_workers: Maximum number of worker threads
            create_indexes: Whether to create performance indexes
            engine: CSV reading engine ("pandas", "polars", or "duckdb")
        """
        self.db_path = Path(db_path) if db_path else None
        self.table_name = table_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.create_indexes = create_indexes
        self.engine = engine
        self._table_created = False
        self._schema = None
        
        # Initialize components
        self.file_reader = CSVFileReader(max_workers=max_workers)
        
        # Initialize database handler
        self.db_handler = DuckDBHandler(db_path=self.db_path)
        
        # Initialize database and metadata tables
        self.db_handler.init_db()
        self.db_handler.init_csv_metadata_tables()
        
        logger.info(f"Initialized CSVDataFeed with database: {self.db_path}, table: {self.table_name}, engine: {self.engine}")
    
    def analyze_schema(self, file_infos: List[FileInfo], sample_size: int = 5) -> Dict:
        """Analyze CSV files to determine schema."""
        logger.info("Analyzing CSV schema for table...")
        
        # Sample files for schema analysis
        sample_files = file_infos[:sample_size] if len(file_infos) > sample_size else file_infos
        
        all_columns = set()
        column_types = {}
        
        for file_info in sample_files:
            try:
                df = self.file_reader.read_csv_data(file_info, engine=self.engine)
                if df.empty:
                    continue
                
                # Clean and formalize dataframe
                df = self._formalize_dataframe(df)
                
                all_columns.update(df.columns)
                
                for col, dtype in df.dtypes.items():
                    if col not in column_types:
                        column_types[col] = []
                    column_types[col].append(str(dtype))
                
            except Exception as e:
                logger.warning(f"Failed to analyze schema for {file_info.filename}: {e}")
        
        # Determine unified types
        unified_types = {}
        for col, types in column_types.items():
            if len(set(types)) > 1:
                unified_types[col] = "object"
            else:
                unified_types[col] = types[0]
        
        # Maintain order based on the column mapping
        ordered_columns = ["datetime", "ticker", "open", "high", "low", "close", "volume", "amount"]
        remaining_columns = sorted(col for col in all_columns if col not in ordered_columns)
        
        schema = {
            "columns": ordered_columns + remaining_columns,
            "column_types": unified_types
        }
        
        logger.info(f"Analyzed schema: {len(all_columns)} unique columns found")
        return schema
    
    def _formalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dataframe format, including column names, data types, and string format."""
        # Column name mapping from Chinese to English
        column_mapping = {
            "日期": "datetime",
            "代码": "ticker", 
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low", 
            "收盘价": "close",
            "成交量（手）": "volume",
            "成交额（元）": "amount",
        }
        
        # Apply mapping
        df_renamed = df.rename(columns=column_mapping)
        
        # Clean any remaining column names
        clean_columns = {}
        for col in df_renamed.columns:
            clean_col = self._clean_column_name(col)
            if clean_col != col:
                clean_columns[col] = clean_col
        
        if clean_columns:
            df_renamed = df_renamed.rename(columns=clean_columns)
        
        # Convert datetime column if present
        if "datetime" in df_renamed.columns:
            try:
                df_renamed["datetime"] = pd.to_datetime(df_renamed["datetime"], format="mixed")
            except Exception as e:
                logger.warning(f"Failed to convert datetime column: {e}")
        
        # Unify ticker format
        if "ticker" in df_renamed.columns:
            df_renamed["ticker"] = self._normalize_ticker_format(df_renamed["ticker"])
        
        return df_renamed
    
    def _normalize_ticker_format(self, ticker_series: pd.Series) -> pd.Series:
        """
        Normalize ticker format to standard format (e.g., 600223.SH, 000001.SZ).
        Uses efficient vectorized operations with conditional logic to minimize processing.
        """
        ticker_col = ticker_series.astype(str)
        
        # Check for different format patterns and apply targeted transformations
        # SH.600223 -> 600223.SH (Shanghai A-shares)
        sh_pattern = ticker_col.str.startswith("SH.")
        if sh_pattern.any():
            ticker_col.loc[sh_pattern] = ticker_col.loc[sh_pattern].str.replace("SH.", "", regex=False) + ".SH"
        
        # SZ.000001 -> 000001.SZ (Shenzhen A-shares) 
        sz_pattern = ticker_col.str.startswith("SZ.")
        if sz_pattern.any():
            ticker_col.loc[sz_pattern] = ticker_col.loc[sz_pattern].str.replace("SZ.", "", regex=False) + ".SZ"
        
        # 600223.XSHG -> 600223.SH (Wind format to standard format)
        xshg_pattern = ticker_col.str.endswith(".XSHG")
        if xshg_pattern.any():
            ticker_col.loc[xshg_pattern] = ticker_col.loc[xshg_pattern].str.replace(".XSHG", ".SH", regex=False)
        
        # 000001.XSHE -> 000001.SZ (Wind format to standard format)
        xshe_pattern = ticker_col.str.endswith(".XSHE")
        if xshe_pattern.any():
            ticker_col.loc[xshe_pattern] = ticker_col.loc[xshe_pattern].str.replace(".XSHE", ".SZ", regex=False)
        
        return ticker_col
    
    def create_table(self, schema: Dict):
        """Creat table with schema that accommodates all CSV files."""
        if self._table_created:
            return
        
        # Data columns from CSV files
        all_columns = []
        dtype_mapping = {
            "object": "VARCHAR",
            "int64": "BIGINT",
            "int32": "INTEGER", 
            "float64": "DOUBLE",
            "float32": "FLOAT",
            "bool": "BOOLEAN",
            "datetime64[ns]": "TIMESTAMP"
        }
        
        # Special handling for known financial data columns
        required_columns = {
            "datetime": "TIMESTAMP",
            "ticker": "VARCHAR",
            "open": "DOUBLE",
            "high": "DOUBLE", 
            "low": "DOUBLE",
            "close": "DOUBLE",
            "volume": "DOUBLE",
            "amount": "DOUBLE",
        }
        
        for col in schema["columns"]:
            clean_col = self._clean_column_name(col)
            
            # Use special column type if defined
            if clean_col in required_columns:
                duck_type = required_columns[clean_col]
            else:
                pandas_type = schema["column_types"].get(col, "object")
                duck_type = dtype_mapping.get(pandas_type, "VARCHAR")
                
                # General date/time detection
                if any(keyword in col.lower() for keyword in ["date", "time", "datetime"]):
                    duck_type = "TIMESTAMP"
            
            all_columns.append(f"{clean_col} {duck_type}")
        
        
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {', '.join(all_columns)},
                PRIMARY KEY (datetime, ticker)
            )
        """
        
        self.db_handler._query(create_sql)
        self._table_created = True
        self._schema = schema
        
        logger.info(f"Created table '{self.table_name}' with {len(all_columns)} columns")
    
    def _clean_column_name(self, col_name: str) -> str:
        """Clean column name for database compatibility."""
        import re
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(col_name))
        if clean_name and clean_name[0].isdigit():
            clean_name = f"col_{clean_name}"
        return clean_name.lower()
    
    def _harmonize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize DataFrame schema to match table."""
        if not self._schema:
            return df
        
        df_harmonized = df.copy()
        
        # Ensure all expected columns exist
        for col in self._schema["columns"]:
            clean_col = self._clean_column_name(col)
            if clean_col not in df_harmonized.columns:
                df_harmonized[clean_col] = None
        
        # Clean column names (except metadata columns)
        column_mapping = {}
        for col in df_harmonized.columns:
            if not col.startswith("_"):
                clean_col = self._clean_column_name(col)
                if clean_col != col:
                    column_mapping[col] = clean_col
        
        if column_mapping:
            df_harmonized = df_harmonized.rename(columns=column_mapping)
        
        return df_harmonized
    
    def discover_files(self, root_path: Union[str, Path]) -> List[FileInfo]:
        """
        Discover all CSV files in the given root path.
        
        Args:
            root_path: Root directory to search for files
            
        Returns:
            List of FileInfo objects
        """
        return self.file_reader.discover_files(root_path)
    
    def load_file(self, file_info: FileInfo, skip_duplicates: bool = True) -> bool:
        """
        Load a single CSV file into DuckDB table.
        
        Args:
            file_info: FileInfo object containing file details
            skip_duplicates: Whether to skip files that have been processed before
            
        Returns:
            True if file was successfully loaded, False otherwise
        """
        try:
            # Check for duplicates
            if skip_duplicates:
                file_hash = self.file_reader.calculate_file_hash(file_info)
                file_path_for_metadata = self._get_metadata_file_path(file_info)
                
                if self.db_handler.check_file_processed(file_path_for_metadata, file_hash):
                    logger.info(f"Skipping already processed file: {file_info.filename}")
                    return True
            
            # Read CSV data
            logger.info(f"Loading file: {file_info.filename}")
            df = self.file_reader.read_csv_data(file_info, engine=self.engine)
            
            if df.empty:
                logger.warning(f"Empty CSV file: {file_info.filename}")
                return False
            
            # Formalize dataframe
            df = self._formalize_dataframe(df)
            
            # Ensure table is created
            if not self._table_created:
                logger.error("Table not created. Please run analyze_schema() and create_table() first.")
                return False
            
            # Harmonize schema to match table
            df_harmonized = self._harmonize_schema(df)
            
            # Insert data in batches
            self.db_handler.batch_insert_dataframe(df_harmonized, self.table_name, self.batch_size)
            
            # Update metadata
            metadata = {
                "file_path": self._get_metadata_file_path(file_info),
                "table_name": self.table_name,
                "year": file_info.year,
                "month": file_info.month,
                "exchange": file_info.exchange,
                "symbol": file_info.symbol,
                "file_size": file_info.file_size,
                "row_count": len(df),
                "processed_at": datetime.now(),
                "file_hash": self.file_reader.calculate_file_hash(file_info) if skip_duplicates else ""
            }
            
            self.db_handler.store_file_metadata(metadata)
            
            logger.info(f"Successfully loaded {len(df)} rows into table {self.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load file {file_info.filename}: {e}")
            return False
    
    def load_directory(self, root_path: Union[str, Path], 
                      skip_duplicates: bool = True, 
                      parallel: bool = True) -> Dict[str, int]:
        """
        Load all CSV files from a directory tree into DuckDB table.
        
        Args:
            root_path: Root directory to process
            skip_duplicates: Whether to skip already processed files
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting directory load: {root_path}")
        
        # Discover files
        discovered_files = self.discover_files(root_path)
        
        if not discovered_files:
            logger.warning("No CSV files found")
            return {"total_files": 0, "successful": 0, "failed": 0}
        
        # Analyze schema and create table if needed
        if not self._table_created:
            schema = self.analyze_schema(discovered_files)
            self.create_table(schema)
        
        # Process files
        stats = {"total_files": len(discovered_files), "successful": 0, "failed": 0}
        
        if parallel and self.max_workers > 1:
            # Use parallel processing through file reader
            results = self.file_reader.process_files_parallel(
                discovered_files, 
                self._load_file_wrapper,
                skip_duplicates=skip_duplicates
            )
            
            for result in results:
                if result:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
        else:
            # Sequential processing
            for file_info in discovered_files:
                success = self.load_file(file_info, skip_duplicates)
                if success:
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
        
        logger.info(f"Directory loading completed: {stats}")
        return stats
    
    def _load_file_wrapper(self, file_info: FileInfo, **kwargs) -> bool:
        """Wrapper method for parallel processing"""
        return self.load_file(file_info, **kwargs)
    
    def _get_metadata_file_path(self, file_info: FileInfo) -> str:
        """Generate consistent file path for metadata storage"""
        if file_info.source_type in ["zip", "7z"]:
            return f"{file_info.container_path}:{file_info.csv_path_in_archive}"
        else:
            return file_info.file_path
    
    def get_loading_summary(self) -> pd.DataFrame:
        """Get summary of loaded files and tables"""
        return self.db_handler.get_csv_loading_summary()
    
    def query_data(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        return self.db_handler._query(query, fmt="dataframe")
    
    def list_tables(self) -> List[str]:
        """List all data tables (excluding metadata tables)"""
        return self.db_handler.list_csv_tables()
    
    def close(self):
        """Close database connection"""
        # DuckDBHandler uses singleton pattern with connection management
        logger.info("CSV loader closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb  # Unused parameters
        self.close()


# Convenience functions for common use cases
def quick_load(source_path: str, db_path: str, **kwargs) -> Dict[str, int]:
    """
    Quick function to load CSV files into DuckDB with default settings.
    
    Args:
        source_path: Path to directory containing CSV files
        db_path: Path to DuckDB database file
        **kwargs: Additional arguments for CSVToDuckDBLoader
        
    Returns:
        Loading statistics
    """
    with CSVDataFeed(db_path, **kwargs) as loader:
        return loader.load_directory(source_path)

