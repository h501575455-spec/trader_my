import pandas as pd
import duckdb
import datetime
import json
import threading
from typing import Optional, Dict, Callable, Any, List

from .base import FactorHandler
from ...expression.base import Factor
from ....data.database import build_config, DatabaseTypes
from ....utils.log import GL


class DuckDBFactorHandler(FactorHandler):
    """DuckDB implementation of FactorHandler"""
    
    _lock = threading.Lock()  # thread-safe lock
    
    def __init__(self):
        self.config = build_config(DatabaseTypes.DUCKDB.value)
        self.db_path = self.config.factor_path
        self._logger = GL.get_logger(__name__)
    
    def init_db(self) -> None:
        """Initialize DuckDB connection"""
        with duckdb.connect(self.db_path) as conn:
            conn.execute("SELECT 1")
    
    def _execute_with_conn(self, operation: Callable[[Any], Any], read_only: bool = False) -> Any:
        """Execute operation with managed DuckDB connection"""
        with self._lock:
            with duckdb.connect(self.db_path, read_only=read_only) as conn:
                return operation(conn)
    
    def factor2db(self, factor: Factor, factor_name: str, table_name: str, **kwargs) -> None:
        """Store factor data to DuckDB"""
        def _operation(conn):
            # Transform and register factor data
            factor_data = self._transform_factor_data(factor, factor_name)
            conn.register("factor_data", factor_data)
            
            # Check table existence and create/update accordingly
            if not self._check_table_exists(conn, table_name):
                self._create_table_and_insert_data(conn, table_name, factor_name)
            else:
                self._update_existing_data(conn, table_name, factor_name)
        
        self._execute_with_conn(_operation)
    
    def read_factor(self, table_name: str, factor_name: str, 
                   start_date: str = None, end_date: str = None) -> Factor:
        """Read factor data from DuckDB"""
        def _operation(conn):
            query = f"SELECT ticker, trade_date, {factor_name} FROM {table_name}"
            params = []
            
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("trade_date >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("trade_date <= ?")
                    params.append(end_date)
                query += " WHERE " + " AND ".join(conditions)
            
            if params:
                data = conn.execute(query, params).fetch_df()
            else:
                data = conn.execute(query).fetch_df()
            
            if data.empty:
                return Factor(pd.DataFrame())
            
            # Transform data back to factor format
            data.set_index(["ticker", "trade_date"], inplace=True)
            data = data.swaplevel().unstack()
            data.columns = data.columns.droplevel(level=0)
            data.sort_index(inplace=True)
            
            return Factor(data)
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def check_factor_exists(self, table_name: str, factor_name: str) -> bool:
        """Check if factor exists in DuckDB table"""
        def _operation(conn):
            # First check if table exists
            table_result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", 
                [table_name]
            ).fetchone()
            
            if not table_result or table_result[0] == 0:
                return False
            
            # Then check if column exists
            column_result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                [table_name, factor_name]
            ).fetchone()
            
            return column_result and column_result[0] > 0
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def get_factor_date_range(self, table_name: str, factor_name: str) -> tuple:
        """Get factor data date range from DuckDB"""
        def _operation(conn):
            result = conn.execute(
                f"SELECT MIN(trade_date), MAX(trade_date) FROM {table_name} WHERE {factor_name} IS NOT NULL"
            ).fetchone()
            
            if result and result[0] is not None:
                return result[0], result[1]
            return None, None
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def delete_factor(self, table_name: str, factor_name: str) -> None:
        """Delete factor column from DuckDB table"""
        def _operation(conn):
            conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {factor_name}")
        
        self._execute_with_conn(_operation)
    
    def _transform_factor_data(self, factor: Factor, factor_name: str) -> pd.DataFrame:
        """Transform factor data for DuckDB storage"""
        factor_data = factor.data.stack()
        factor_data.name = factor_name
        factor_data.index.names = ["trade_date", "ticker"]
        factor_data = factor_data.swaplevel().sort_index().reset_index()
        return factor_data
    
    def _check_table_exists(self, conn, table_name: str) -> bool:
        """Check if table exists"""
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", 
            [table_name]
        ).fetchone()
        return result[0] > 0
    
    def _create_table_and_insert_data(self, conn, table_name: str, factor_name: str):
        """Create new table and insert data"""
        conn.execute(f"""
            CREATE TABLE {table_name} (
                ticker VARCHAR,
                trade_date DATE,
                {factor_name} DOUBLE,
                PRIMARY KEY (ticker, trade_date)
            )
        """)
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM factor_data")
        self._logger.info(f"Table '{table_name}' created and initial factor data inserted.")
    
    def _update_existing_data(self, conn, table_name: str, factor_name: str):
        """Update existing table with new data"""
        # Check if column exists
        column_result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
            [table_name, factor_name]
        ).fetchone()
        
        if column_result[0] == 0:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {factor_name} DOUBLE")
            self._logger.info(f"Column '{factor_name}' added.")
        
        # Upsert data
        conn.execute(f"CREATE TEMPORARY TABLE factor_temp AS SELECT * FROM factor_data")
        conn.execute(f"""
            INSERT INTO {table_name} (ticker, trade_date, {factor_name})
            SELECT ticker, trade_date, {factor_name} FROM factor_temp
            ON CONFLICT (ticker, trade_date) 
            DO UPDATE SET {factor_name} = excluded.{factor_name}
        """)
        conn.execute("DROP TABLE factor_temp")
        self._logger.info(f"Factor '{factor_name}' data updated.")
    
    # Factor metadata management methods
    def save_factor_metadata(self, metadata) -> None:
        """Save factor metadata to DuckDB"""
        def _operation(conn):
            # Create metadata table if not exists
            if not self._check_metadata_table_exists(conn):
                conn.execute("""
                    CREATE TABLE factor_metadata (
                        uid VARCHAR PRIMARY KEY,
                        name VARCHAR UNIQUE NOT NULL,
                        description VARCHAR,
                        dependencies VARCHAR,  -- JSON string
                        category VARCHAR,
                        tags VARCHAR,  -- JSON string
                        author VARCHAR,
                        lifecycle_status VARCHAR DEFAULT 'development',
                        version VARCHAR DEFAULT '1.0.0',
                        created_time TIMESTAMP,
                        updated_time TIMESTAMP
                    )
                """)
            else:
                # Check if table has uid column, if not add it
                try:
                    conn.execute("SELECT uid FROM factor_metadata LIMIT 1")
                except:
                    # Add uid column if it doesn't exist
                    conn.execute("ALTER TABLE factor_metadata ADD COLUMN uid VARCHAR")
                    conn.execute("ALTER TABLE factor_metadata ADD COLUMN author VARCHAR")
                    conn.execute("ALTER TABLE factor_metadata ADD COLUMN lifecycle_status VARCHAR DEFAULT 'development'")
                    conn.execute("ALTER TABLE factor_metadata ADD COLUMN version VARCHAR DEFAULT '1.0.0'")
                    # Update existing records to have UIDs - this will be handled by the loading logic
            
            # Convert metadata to values
            values = (
                metadata.uid,
                metadata.name,
                metadata.description,
                json.dumps(metadata.dependencies),
                metadata.category,
                json.dumps(metadata.tags),
                getattr(metadata, "author", ""),
                getattr(metadata, "lifecycle_status", "development").value if hasattr(getattr(metadata, "lifecycle_status", "development"), 'value') else str(getattr(metadata, "lifecycle_status", "development")),
                str(getattr(metadata, "version", "1.0.0")),
                metadata.created_time,
                datetime.datetime.now()
            )
            
            # Upsert metadata
            conn.execute("""
                INSERT INTO factor_metadata
                (uid, name, description, dependencies, category, tags, author, lifecycle_status, version, created_time, updated_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (uid) DO UPDATE SET
                    description = excluded.description,
                    dependencies = excluded.dependencies,
                    category = excluded.category,
                    tags = excluded.tags,
                    author = excluded.author,
                    lifecycle_status = excluded.lifecycle_status,
                    version = excluded.version,
                    updated_time = excluded.updated_time
            """, values)
        
        self._execute_with_conn(_operation)
    
    def load_factor_metadata(self, factor_name: str):
        """Load factor metadata from DuckDB"""
        def _operation(conn):
            if not self._check_metadata_table_exists(conn):
                return None

            result = conn.execute(
                "SELECT uid, name, description, dependencies, category, tags, author, lifecycle_status, version, created_time, updated_time FROM factor_metadata WHERE name = ?",
                [factor_name]
            ).fetchone()
            
            if result:
                # Import here to avoid circular import
                from ..inventory import FactorMetadata
                from ..lifecycle import FactorLifecycleStatus, FactorVersion
                
                # Convert lifecycle_status string to enum
                lifecycle_status_value = result[7] or "development"
                try:
                    lifecycle_status = FactorLifecycleStatus(lifecycle_status_value)
                except (ValueError, TypeError):
                    lifecycle_status = FactorLifecycleStatus.DEVELOPMENT

                # Convert version string to FactorVersion object
                version_str = result[8] or "1.0.0"
                try:
                    version = FactorVersion.from_string(version_str) if hasattr(FactorVersion, 'from_string') else FactorVersion()
                except:
                    version = FactorVersion()

                return FactorMetadata(
                    uid=result[0] or "",  # uid
                    name=result[1],       # name
                    description=result[2] or "",
                    dependencies=json.loads(result[3]) if result[3] else [],
                    category=result[4] or "default",
                    tags=json.loads(result[5]) if result[5] else [],
                    author=result[6] or "",
                    lifecycle_status=lifecycle_status,
                    version=version,
                    created_time=result[9],
                    updated_time=result[10]
                )
            return None
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def load_all_factor_metadata(self) -> Dict[str, any]:
        """Load all factor metadata from DuckDB"""
        def _operation(conn):
            if not self._check_metadata_table_exists(conn):
                return {}
            
            results = conn.execute("SELECT uid, name, description, dependencies, category, tags, author, lifecycle_status, version, created_time, updated_time FROM factor_metadata").fetchall()

            metadata_dict = {}
            for result in results:
                # Import here to avoid circular import
                from ..inventory import FactorMetadata
                from ..lifecycle import FactorLifecycleStatus, FactorVersion

                # Convert lifecycle_status string to enum
                lifecycle_status_value = result[7] or "development"
                try:
                    lifecycle_status = FactorLifecycleStatus(lifecycle_status_value)
                except (ValueError, TypeError):
                    lifecycle_status = FactorLifecycleStatus.DEVELOPMENT

                # Convert version string to FactorVersion object
                version_str = result[8] or "1.0.0"
                try:
                    version = FactorVersion.from_string(version_str) if hasattr(FactorVersion, 'from_string') else FactorVersion()
                except:
                    version = FactorVersion()

                metadata = FactorMetadata(
                    uid=result[0] or "",  # uid
                    name=result[1],       # name
                    description=result[2] or "",
                    dependencies=json.loads(result[3]) if result[3] else [],
                    category=result[4] or "default",
                    tags=json.loads(result[5]) if result[5] else [],
                    author=result[6] or "",
                    lifecycle_status=lifecycle_status,
                    version=version,
                    created_time=result[9],
                    updated_time=result[10]
                )
                # Use UID as key instead of name for the new format
                metadata_dict[result[0] or result[1]] = metadata
            
            return metadata_dict
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def delete_factor_metadata(self, factor_name: str) -> None:
        """Delete factor metadata from DuckDB"""
        def _operation(conn):
            if self._check_metadata_table_exists(conn):
                conn.execute(
                    "DELETE FROM factor_metadata WHERE name = ?", 
                    [factor_name]
                )
        
        self._execute_with_conn(_operation)
    
    def check_metadata_table_exists(self) -> bool:
        """Check if metadata table exists in DuckDB"""
        def _operation(conn):
            return self._check_metadata_table_exists(conn)
        
        return self._execute_with_conn(_operation, read_only=True)
    
    def _check_metadata_table_exists(self, conn) -> bool:
        """Internal method to check if metadata table exists"""
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'factor_metadata'"
        ).fetchone()
        return result[0] > 0
    
    def close(self):
        """Close connection - No-op for connection-per-operation pattern"""
        pass

    # Lifecycle management methods
    def _ensure_lifecycle_tables(self) -> None:
        """Ensure lifecycle metadata tables exist"""
        def _operation(conn):
            create_lifecycle_table_sql = """
            CREATE TABLE IF NOT EXISTS factor_lifecycle (
                factor_name VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                version_major INTEGER NOT NULL DEFAULT 1,
                version_minor INTEGER NOT NULL DEFAULT 0,
                version_patch INTEGER NOT NULL DEFAULT 0,
                lifecycle_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            conn.execute(create_lifecycle_table_sql)
            conn.commit()
        
        self._execute_with_conn(_operation)
    
    def save_lifecycle_metadata(self, factor_name: str, lifecycle_meta) -> bool:
        """Save lifecycle metadata to DuckDB"""
        try:
            # Ensure lifecycle tables exist before saving
            self._ensure_lifecycle_tables()
            
            def _operation(conn):
                data = self._serialize_lifecycle_for_db(lifecycle_meta)
                lifecycle_json = json.dumps(data)
                
                sql = """
                INSERT OR REPLACE INTO factor_lifecycle 
                (factor_name, status, version_major, version_minor, version_patch, lifecycle_data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """
                
                conn.execute(sql, [
                    factor_name,
                    lifecycle_meta.status.value,
                    lifecycle_meta.version.major,
                    lifecycle_meta.version.minor,
                    lifecycle_meta.version.patch,
                    lifecycle_json
                ])
                conn.commit()
            
            self._execute_with_conn(_operation)
            self._logger.info(f"Saved lifecycle metadata for factor '{factor_name}'")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save lifecycle metadata for '{factor_name}': {e}")
            return False
    
    def load_lifecycle_metadata(self, factor_name: str):
        """Load lifecycle metadata from DuckDB"""
        # Ensure lifecycle tables exist before loading
        self._ensure_lifecycle_tables()
        try:
            def _operation(conn):
                sql = "SELECT lifecycle_data FROM factor_lifecycle WHERE factor_name = ?"
                result = conn.execute(sql, [factor_name]).fetchone()
                
                if result:
                    data = json.loads(result[0])
                    return self._deserialize_lifecycle_from_db(data)
                return None
            
            return self._execute_with_conn(_operation, read_only=True)
            
        except Exception as e:
            self._logger.error(f"Failed to load lifecycle metadata for '{factor_name}': {e}")
            return None
    
    def load_all_lifecycle_metadata(self) -> Dict[str, Any]:
        """Load all lifecycle metadata from DuckDB"""
        try:
            def _operation(conn):
                sql = "SELECT factor_name, lifecycle_data FROM factor_lifecycle"
                results = conn.execute(sql).fetchall()
                
                metadata_dict = {}
                for factor_name, lifecycle_json in results:
                    data = json.loads(lifecycle_json)
                    lifecycle_meta = self._deserialize_lifecycle_from_db(data)
                    metadata_dict[factor_name] = lifecycle_meta
                
                return metadata_dict
            
            return self._execute_with_conn(_operation, read_only=True)
            
        except Exception as e:
            self._logger.error(f"Failed to load all lifecycle metadata: {e}")
            return {}
    
    def delete_lifecycle_metadata(self, factor_name: str) -> bool:
        """Delete lifecycle metadata from DuckDB"""
        try:
            def _operation(conn):
                sql = "DELETE FROM factor_lifecycle WHERE factor_name = ?"
                result = conn.execute(sql, [factor_name])
                conn.commit()
                return result.rowcount > 0
            
            success = self._execute_with_conn(_operation)
            if success:
                self._logger.info(f"Deleted lifecycle metadata for factor '{factor_name}'")
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to delete lifecycle metadata for '{factor_name}': {e}")
            return False
    
    def query_factors_by_status(self, status) -> List[str]:
        """Query factors by lifecycle status"""
        try:
            def _operation(conn):
                sql = "SELECT factor_name FROM factor_lifecycle WHERE status = ?"
                results = conn.execute(sql, [status.value]).fetchall()
                return [row[0] for row in results]
            
            return self._execute_with_conn(_operation, read_only=True)
            
        except Exception as e:
            self._logger.error(f"Failed to query factors by status '{status.value}': {e}")
            return []
    
    def get_lifecycle_history(self, factor_name: str, limit: int = 100) -> List[Dict]:
        """Get lifecycle transition history for a factor"""
        try:
            lifecycle_meta = self.load_lifecycle_metadata(factor_name)
            if lifecycle_meta:
                # Sort by timestamp and limit results
                history = sorted(lifecycle_meta.status_history, 
                               key=lambda x: x.timestamp, reverse=True)
                limited_history = history[:limit]
                
                return [{
                    "from_status": record.from_status.value,
                    "to_status": record.to_status.value,
                    "timestamp": record.timestamp.isoformat(),
                    "operator": record.operator,
                    "reason": record.reason,
                    "metadata": record.metadata
                } for record in limited_history]
            
            return []
            
        except Exception as e:
            self._logger.error(f"Failed to get lifecycle history for '{factor_name}': {e}")
            return []