#!/usr/bin/env python3
"""
CSV File Reader

Handles CSV file discovery, reading, and processing from various sources
including hierarchical folder structures, zip archives, and 7z archives.

Supported structures:
1. 2024年.zip -> 01.zip, 02.zip -> ticker.csv
2. 2023年.7z -> ticker.csv (direct)
3. Loose files in directories

Separated from database operations for better modularity.
"""

import re
import hashlib
import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import py7zr
    HAS_7Z_SUPPORT = True
except ImportError:
    HAS_7Z_SUPPORT = False

try:
    import polars as pl
    HAS_POLARS_SUPPORT = True
except ImportError:
    HAS_POLARS_SUPPORT = False

try:
    import duckdb
    HAS_DUCKDB_SUPPORT = True
except ImportError:
    HAS_DUCKDB_SUPPORT = False

from ...utils.log import L

logger = L.get_logger("frozen")


@dataclass
class FileInfo:
    """Information about discovered CSV files"""
    file_path: str
    filename: str
    source_type: str  # 'zip', '7z', or 'directory'
    year: Optional[str] = None
    month: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    file_size: int = 0
    container_path: Optional[str] = None  # For zip/7z files
    csv_path_in_archive: Optional[str] = None  # For zip/7z files
    nested_archive: Optional[str] = None  # For nested archives (year.zip -> month.zip)


class CSVFileReader:
    """
    CSV file discovery and reading functionality.
    
    Features:
    - Hierarchical file structure discovery with multiple archive formats
    - Support for nested archives (year.zip -> month.zip -> csv)
    - Support for direct archives (year.7z -> csv)
    - Pattern-based file parsing and metadata extraction
    - Support for loose files and multiple archive formats
    - Parallel file processing capabilities
    - File hash calculation for duplicate detection
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the CSV file reader.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = max_workers
        
        # File pattern matching
        self.file_patterns = {
            "year_archive": re.compile(r"(\d{4})年\.(zip|7z)"),     # 2024年.zip or 2023年.7z
            "month_zip": re.compile(r"(\d{2})\.zip"),              # 01.zip, 02.zip  
            "csv_pattern": re.compile(r"([A-Z]{2})\.(\d{6})\.csv"), # SZ.002605.csv
            "csv_with_path": re.compile(r"(?:.*/)?([A-Z]{2})\.(\d{6})\.csv"), # Support files in subdirectories
            "general_csv": re.compile(r".*\.csv$", re.IGNORECASE)
        }
        
        if not HAS_7Z_SUPPORT:
            logger.warning("py7zr not installed - 7z file support disabled. Install with: pip install py7zr")
        
        if not HAS_POLARS_SUPPORT:
            logger.warning("polars not installed - Polars engine support disabled. Install with: pip install polars")
        
        if not HAS_DUCKDB_SUPPORT:
            logger.warning("duckdb not installed - DuckDB engine support disabled. Install with: pip install duckdb")
        
        logger.info(f"Initialized CSVFileReader with {max_workers} workers")
    
    def discover_files(self, root_path: Union[str, Path]) -> List[FileInfo]:
        """
        Discover all CSV files in the given root path with support for:
        1. 年份.zip -> 月份.zip -> CSV文件 (nested structure)
        2. 年份.7z -> CSV文件 (direct structure)
        3. Loose files in directories
        
        Args:
            root_path: Root directory to search for files
            
        Returns:
            List of FileInfo objects
        """
        root_path = Path(root_path)
        discovered_files = []
        
        if not root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        
        logger.info(f"Discovering files in: {root_path}")
        
        # Process year archives in root directory
        for archive_file in root_path.iterdir():
            if archive_file.is_file():
                year_match = self.file_patterns["year_archive"].match(archive_file.name)
                if year_match:
                    year = year_match.group(1)
                    archive_type = year_match.group(2)
                    
                    logger.info(f"Processing year archive: {archive_file.name}")
                    
                    try:
                        if archive_type == "zip":
                            # Handle nested structure: year.zip -> month.zip -> csv
                            archive_files = self._discover_files_in_nested_zip(archive_file, year)
                        elif archive_type == "7z" and HAS_7Z_SUPPORT:
                            # Handle direct structure: year.7z -> csv
                            archive_files = self._discover_files_in_7z(archive_file, year)
                        else:
                            logger.warning(f"Unsupported archive type or missing support: {archive_type}")
                            continue
                            
                        discovered_files.extend(archive_files)
                    except Exception as e:
                        logger.warning(f"Failed to process archive {archive_file}: {e}")
        
        # Process any subdirectories for year folders (legacy support)
        for year_folder in root_path.iterdir():
            if year_folder.is_dir():
                # Extract year from folder name (e.g., "2024年" -> "2024")
                year_match = re.match(r"(\d{4})年?", year_folder.name)
                if year_match:
                    year = year_match.group(1)
                    logger.info(f"Processing year folder: {year_folder.name}")
                    
                    # Process month zip files in year folder
                    for month_zip in year_folder.glob("*.zip"):
                        month_match = self.file_patterns["month_zip"].match(month_zip.name)
                        if month_match:
                            month = month_match.group(1)
                            logger.info(f"  Processing month zip: {month_zip.name}")
                            
                            try:
                                zip_files = self._discover_files_in_zip(month_zip, year, month)
                                discovered_files.extend(zip_files)
                            except Exception as e:
                                logger.warning(f"Failed to process ZIP file {month_zip}: {e}")
                    
                    # Also process loose CSV files in year folders
                    loose_files = self._discover_loose_files(year_folder, year)
                    discovered_files.extend(loose_files)
        
        # Also process any loose files in root directory
        root_loose_files = self._discover_loose_files(root_path)
        discovered_files.extend(root_loose_files)
        
        logger.info(f"Discovered {len(discovered_files)} CSV files")
        return discovered_files
    
    def _discover_files_in_nested_zip(self, year_zip_path: Path, year: str) -> List[FileInfo]:
        """
        Discover CSV files in nested ZIP structure: year.zip -> month.zip -> csv
        """
        files = []
        
        with zipfile.ZipFile(year_zip_path, "r") as year_zf:
            # Look for month zip files inside year zip
            for zip_filename in year_zf.namelist():
                # Skip system files
                if "__MACOSX" in zip_filename or zip_filename.startswith("."):
                    continue
                    
                if zip_filename.lower().endswith(".zip"):
                    month_match = self.file_patterns["month_zip"].match(Path(zip_filename).name)
                    if month_match:
                        month = month_match.group(1)
                        logger.info(f"    Processing nested month zip: {zip_filename}")
                        
                        # Extract the month zip file to memory and process it
                        month_zip_data = year_zf.read(zip_filename)
                        
                        # Create a temporary file-like object
                        import io
                        month_zip_io = io.BytesIO(month_zip_data)
                        
                        try:
                            with zipfile.ZipFile(month_zip_io, "r") as month_zf:
                                for csv_filename in month_zf.namelist():
                                    # Skip system files and non-CSV files
                                    if ("__MACOSX" in csv_filename or 
                                        csv_filename.startswith(".") or 
                                        not csv_filename.lower().endswith(".csv")):
                                        continue
                                        
                                    file_info = self._parse_csv_filename(csv_filename)
                                    
                                    # Create FileInfo object for nested archive
                                    files.append(FileInfo(
                                        file_path=f"{year_zip_path}:{zip_filename}:{csv_filename}",
                                        filename=csv_filename,
                                        source_type="zip",
                                        year=year,
                                        month=month,
                                        exchange=file_info.get("exchange"),
                                        symbol=file_info.get("symbol"),
                                        file_size=month_zf.getinfo(csv_filename).file_size,
                                        container_path=str(year_zip_path),
                                        csv_path_in_archive=csv_filename,
                                        nested_archive=zip_filename
                                    ))
                        except Exception as e:
                            logger.warning(f"Failed to process nested ZIP {zip_filename}: {e}")
        
        return files
    
    def _discover_files_in_7z(self, sevenz_path: Path, year: str) -> List[FileInfo]:
        """
        Discover CSV files in 7z archive: year.7z -> csv (direct)
        """
        files = []
        
        if not HAS_7Z_SUPPORT:
            logger.warning(f"py7zr not available, skipping 7z file: {sevenz_path}")
            return files
        
        try:
            with py7zr.SevenZipFile(sevenz_path, mode="r") as sevenz_f:
                # Get list of files in the 7z archive
                file_list = sevenz_f.list()
                
                for file_info_7z in file_list:
                    filename = file_info_7z.filename
                    
                    # Skip system files and non-CSV files
                    if ("__MACOSX" in filename or 
                        filename.startswith(".") or 
                        not filename.lower().endswith(".csv")):
                        continue
                        
                    file_info = self._parse_csv_filename(filename)
                    
                    # Create FileInfo object
                    files.append(FileInfo(
                        file_path=f"{sevenz_path}:{filename}",
                        filename=filename,
                        source_type="7z",
                        year=year,
                        month=None,  # 7z files don't have month structure
                        exchange=file_info.get("exchange"),
                        symbol=file_info.get("symbol"),
                        file_size=file_info_7z.uncompressed if hasattr(file_info_7z, "uncompressed") else 0,
                        container_path=str(sevenz_path),
                        csv_path_in_archive=filename
                    ))
        except Exception as e:
            logger.error(f"Failed to process 7z file {sevenz_path}: {e}")
        
        return files
    
    def _discover_files_in_zip(self, zip_path: Path, year: str, month: str) -> List[FileInfo]:
        """Discover CSV files within a zip archive (original method for backward compatibility)"""
        files = []
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            for csv_filename in zf.namelist():
                # Skip system files and non-CSV files
                if ("__MACOSX" in csv_filename or 
                    csv_filename.startswith(".") or 
                    not csv_filename.lower().endswith(".csv")):
                    continue
                    
                file_info = self._parse_csv_filename(csv_filename)
                
                # Create FileInfo object
                files.append(FileInfo(
                    file_path=f"{zip_path}:{csv_filename}",  # Unique identifier
                    filename=csv_filename,
                    source_type="zip",
                    year=year,
                    month=month,
                    exchange=file_info.get("exchange"),
                    symbol=file_info.get("symbol"),
                    file_size=zf.getinfo(csv_filename).file_size,
                    container_path=str(zip_path),
                    csv_path_in_archive=csv_filename
                ))
        
        return files
    
    def _discover_loose_files(self, directory: Path, year: str = None) -> List[FileInfo]:
        """Discover loose CSV files in a directory"""
        files = []
        
        for csv_file in directory.rglob("*.csv"):
            file_info_dict = self._parse_file_path(csv_file)
            
            # Extract month from parent folder if it's a month folder
            month = None
            if year and csv_file.parent != directory:
                month_match = self.file_patterns["month_zip"].match(csv_file.parent.name)
                if month_match:
                    month = month_match.group(1)
            
            # Create FileInfo object
            files.append(FileInfo(
                file_path=str(csv_file),
                filename=csv_file.name,
                source_type="directory",
                year=year,
                month=month,
                exchange=file_info_dict.get("exchange"),
                symbol=file_info_dict.get("symbol"),
                file_size=file_info_dict.get("file_size", 0)
            ))
        
        return files
    
    def _parse_file_path(self, file_path: Path) -> Dict:
        """Parse file path to extract metadata"""
        file_info = {
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
        }
        
        # Try to extract info from filename
        csv_match = self.file_patterns["csv_pattern"].match(file_path.name)
        if csv_match:
            file_info["exchange"] = csv_match.group(1)
            file_info["symbol"] = csv_match.group(2)
        
        return file_info
    
    def _parse_csv_filename(self, filename: str) -> Dict:
        """Parse CSV filename to extract exchange and symbol"""
        # Try the pattern with path support first
        match = self.file_patterns["csv_with_path"].match(filename)
        if match:
            return {
                "exchange": match.group(1),
                "symbol": match.group(2)
            }
        
        # Fallback to original pattern
        match = self.file_patterns["csv_pattern"].match(filename)
        if match:
            return {
                "exchange": match.group(1),
                "symbol": match.group(2)
            }
        return {}
    
    def read_csv_data(self, file_info: FileInfo, engine: str = "pandas") -> pd.DataFrame:
        """
        Read CSV data from file, zip archive, or 7z archive.
        
        Args:
            file_info: FileInfo object containing file details
            engine: CSV reading engine to use ("pandas", "polars", or "duckdb")
            
        Returns:
            pandas DataFrame containing CSV data
        """
        if engine not in ["pandas", "polars", "duckdb"]:
            raise ValueError(f"Unsupported engine: {engine}. Supported engines: 'pandas', 'polars', 'duckdb'")
        
        if engine == "polars" and not HAS_POLARS_SUPPORT:
            logger.warning("Polars not available, falling back to pandas")
            engine = "pandas"
        
        if engine == "duckdb" and not HAS_DUCKDB_SUPPORT:
            logger.warning("DuckDB not available, falling back to pandas")
            engine = "pandas"
        
        if file_info.source_type == "directory":
            data = self._read_csv_with_engine(file_info.file_path, engine, source_type="file")
            return self._drop_first_column(data, engine)
        
        elif file_info.source_type == "zip":
            if file_info.nested_archive:
                # Handle nested zip structure: year.zip -> month.zip -> csv
                with zipfile.ZipFile(file_info.container_path, "r") as year_zf:
                    month_zip_data = year_zf.read(file_info.nested_archive)
                    
                    import io
                    month_zip_io = io.BytesIO(month_zip_data)
                    
                    with zipfile.ZipFile(month_zip_io, "r") as month_zf:
                        with month_zf.open(file_info.csv_path_in_archive) as csv_file:
                            data = self._read_csv_with_engine(csv_file, engine, source_type="buffer")
                            return self._drop_first_column(data, engine)
            else:
                # Handle simple zip structure
                with zipfile.ZipFile(file_info.container_path, "r") as zf:
                    with zf.open(file_info.csv_path_in_archive) as csv_file:
                        data = self._read_csv_with_engine(csv_file, engine, source_type="buffer")
                        return self._drop_first_column(data, engine)
        
        elif file_info.source_type == "7z":
            if not HAS_7Z_SUPPORT:
                raise ValueError("py7zr not installed - cannot read 7z files")
            
            # Handle 7z structure: year.7z -> csv
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with py7zr.SevenZipFile(file_info.container_path, mode="r") as sevenz_f:
                    # Extract the specific file to temp directory
                    sevenz_f.extract(temp_dir, targets=[file_info.csv_path_in_archive])
                    
                    # Read the extracted file
                    extracted_file_path = Path(temp_dir) / file_info.csv_path_in_archive
                    data = self._read_csv_with_engine(extracted_file_path, engine, source_type="file")
                    return self._drop_first_column(data, engine)
        
        else:
            raise ValueError(f"Unsupported source type: {file_info.source_type}")
    
    def _read_csv_with_engine(self, source, engine: str, source_type: str):
        """
        Read CSV data with specified engine.
        
        Args:
            source: File path, file-like object, or buffer
            engine: Engine to use ("pandas", "polars", or "duckdb")
            source_type: Type of source ("file" or "buffer")
            
        Returns:
            DataFrame (pandas or polars depending on engine)
        """
        if engine == "pandas":
            return pd.read_csv(source)
        
        elif engine == "polars":
            if source_type == "file":
                return pl.read_csv(source).to_pandas()
            else:
                # For file-like objects/buffers, we need to use pandas as polars doesn't support them directly
                logger.debug("Polars doesn't support file-like objects, using pandas for this read")
                return pd.read_csv(source)
        
        elif engine == "duckdb":
            if source_type == "file":
                # DuckDB can read CSV files directly and return as pandas DataFrame
                conn = duckdb.connect()
                try:
                    # Use DuckDB's read_csv function which is typically faster than pandas
                    result = conn.execute(f"SELECT * FROM read_csv_auto('{source}')").fetchdf()
                    return result
                finally:
                    conn.close()
            else:
                # For file-like objects/buffers, fall back to pandas
                logger.debug("DuckDB doesn't support file-like objects, using pandas for this read")
                return pd.read_csv(source)
        
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    def _drop_first_column(self, data, engine: str) -> pd.DataFrame:
        """
        Drop the first column from the DataFrame.
        
        Args:
            data: DataFrame (pandas, polars, or duckdb result)
            engine: Engine used to read the data
            
        Returns:
            pandas DataFrame with first column dropped
        """
        # Convert to pandas if needed and drop first column
        if engine == "polars" and hasattr(data, "to_pandas"):
            # If it's still a polars DataFrame, convert it
            data = data.to_pandas()
        elif engine == "duckdb":
            # DuckDB already returns pandas DataFrame
            pass
        
        if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
            return data.drop(data.columns[0], axis=1)
        
        return data
    
    def calculate_file_hash(self, file_info: FileInfo) -> str:
        """Calculate hash for file content to detect changes"""
        hasher = hashlib.md5()
        
        if file_info.source_type == "directory":
            with open(file_info.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        
        elif file_info.source_type == "zip":
            if file_info.nested_archive:
                # Handle nested zip structure
                with zipfile.ZipFile(file_info.container_path, "r") as year_zf:
                    month_zip_data = year_zf.read(file_info.nested_archive)
                    
                    import io
                    month_zip_io = io.BytesIO(month_zip_data)
                    
                    with zipfile.ZipFile(month_zip_io, "r") as month_zf:
                        with month_zf.open(file_info.csv_path_in_archive) as csv_file:
                            for chunk in iter(lambda: csv_file.read(4096), b""):
                                hasher.update(chunk)
            else:
                # Handle simple zip structure
                with zipfile.ZipFile(file_info.container_path, "r") as zf:
                    with zf.open(file_info.csv_path_in_archive) as csv_file:
                        for chunk in iter(lambda: csv_file.read(4096), b""):
                            hasher.update(chunk)
        
        elif file_info.source_type == "7z":
            if not HAS_7Z_SUPPORT:
                raise ValueError("py7zr not installed - cannot hash 7z files")
            
            # Handle 7z structure
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with py7zr.SevenZipFile(file_info.container_path, mode="r") as sevenz_f:
                    # Extract the specific file to temp directory
                    sevenz_f.extract(temp_dir, targets=[file_info.csv_path_in_archive])
                    
                    # Hash the extracted file
                    extracted_file_path = Path(temp_dir) / file_info.csv_path_in_archive
                    with open(extracted_file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def generate_table_name(self, file_info: FileInfo) -> str:
        """Generate appropriate table name based on file information"""
        parts = []
        
        # Add year and month if available
        if file_info.year and file_info.month:
            parts.append(f"data_{file_info.year}_{file_info.month.zfill(2)}")
        
        # Add exchange and symbol if available
        if file_info.exchange and file_info.symbol:
            parts.append(f"{file_info.exchange.lower()}_{file_info.symbol}")
        
        # Fallback to filename-based naming
        if not parts:
            filename = Path(file_info.filename).stem
            # Clean filename for table name
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", filename).lower()
            parts.append(clean_name)
        
        return "_".join(parts)
    
    def process_files_parallel(self, file_infos: List[FileInfo], 
                             processor_func, **kwargs) -> List:
        """Process files in parallel using the provided processor function"""
        results = []
        
        if self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(processor_func, file_info, **kwargs): file_info 
                    for file_info in file_infos
                }
                
                for future in as_completed(future_to_file):
                    file_info = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Parallel processing error for {file_info.filename}: {e}")
                        results.append(None)
        else:
            # Sequential processing
            for file_info in file_infos:
                try:
                    result = processor_func(file_info, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Processing error for {file_info.filename}: {e}")
                    results.append(None)
        
        return results


# Convenience functions
def discover_csv_files(root_path: Union[str, Path], max_workers: int = 4) -> List[FileInfo]:
    """Convenience function to discover CSV files"""
    reader = CSVFileReader(max_workers=max_workers)
    return reader.discover_files(root_path)


def read_csv_file(file_info: FileInfo, engine: str = "pandas") -> pd.DataFrame:
    """Convenience function to read a CSV file with specified engine"""
    reader = CSVFileReader()
    return reader.read_csv_data(file_info, engine=engine)