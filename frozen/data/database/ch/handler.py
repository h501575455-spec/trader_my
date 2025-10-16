import sys
if sys.platform == "darwin" or sys.platform.startswith("linux"):
    import chdb
    from chdb.session import Session

import logging
from typing import Union, Dict

from ..base import DatabaseHandler
from .. import build_config, DatabaseTypes

logger = logging.getLogger("frozen")

class ChdbHandler(DatabaseHandler):
    """ClickHouse database handler"""
    
    def __init__(self):
        self.config = build_config(DatabaseTypes.CHDB.value)
    
    def init_db(self):
        self.config.connect()
        storage_path = self.config.data_path
        self.db = Session(path=storage_path)
        self._query("CREATE DATABASE IF NOT EXISTS BaseDB")
        self._query(("USE BaseDB"))

    def _check_table_exists(self, table_name=""):
        query_str = f"""
                    SELECT count() AS count
                    FROM system.tables 
                    WHERE database='BaseDB' AND name='{table_name}'
                    """
        result = self._query(query_str, "Arrow")
        result_df = chdb.to_df(result)
        count = result_df["count"].iloc[0]
        res = count > 0
        return res
    
    def _insert_df_to_table(self, df, table_name):
        # Convert DataFrame to list of tuples
        records = [tuple(x) for x in df.to_numpy()]
        # format_records = ", ".join(str(tuple(row)) for row in records)
        format_records = ", ".join(
                                f"({','.join(['NULL' if v is None else repr(v) for v in record])})"
                                for record in records
                            )
        # Get columns from DataFrame
        columns = ", ".join(df.columns)
        # Insert data into database
        self._query(f"""
                    INSERT INTO {table_name} ({columns}) 
                    VALUES {format_records};
                    """)
    
    def _delete_table(self, table_name):
        try:
            self._query(f"DROP TABLE IF EXISTS {table_name}")
            logger.critical(f"Table {table_name} has been dropped.")
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")

    def _query(self, query_str: Union[str, Dict], fmt=None):
        """
        Execute query and return results in specified format.
        
        Args:
            query_str: SQL query string
            fmt: Return format
                - "CSV" (default) or "Arrow"
        
        Returns:
            Query results in specified format or None for non-SELECT queries
        """

        if fmt is None:
            fmt = "CSV"
        if fmt not in ["CSV", "Arrow"]:
            raise ValueError(f"chdb only supports 'CSV' or 'Arrow' format, got '{fmt}'")
        return self.db.query(query_str, fmt=fmt)

    def create_volume_price_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker String,
                        trade_date DateTime,
                        close Float64,
                        open Float64,
                        high Float64,
                        low Float64,
                        pre_close Float64,
                        change Float64,
                        pct_chg Float64,
                        volume Float64,
                        amount Float64,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    ENGINE = ReplacingMergeTree
                    ORDER BY (ticker, trade_date);
                    """)
    
    def create_stock_limit_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        trade_date DateTime,
                        ticker String,
                        up_limit Float64,
                        down_limit Float64,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    ENGINE = ReplacingMergeTree
                    ORDER BY (ticker, trade_date);
                    """)
    
    def create_stock_fundamental_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker String,
                        trade_date DateTime,
                        turnover_rate Float64,
                        volume_ratio Float64,
                        pe Float64,
                        pe_ttm Float64,
                        pb Float64,
                        ps Float64,
                        ps_ttm Float64,
                        dv_ratio Float64,
                        dv_ttm Float64,
                        total_share Float64,
                        float_share Float64,
                        total_mv Float64,
                        circ_mv Float64,
                        PRIMARY KEY (ticker, trade_date)
                    )
                    ENGINE = ReplacingMergeTree
                    ORDER BY (ticker, trade_date);
                    """)
    
    def create_stock_dividend_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker String,
                        stk_div Float64,
                        stk_bo_rate Float64,
                        stk_co_rate Float64,
                        cash_div Float64,
                        ex_date DateTime,
                        PRIMARY KEY (ticker, ex_date)
                    )
                    ENGINE = MergeTree
                    ORDER BY (ticker, ex_date);
                    """)
    
    def create_stock_suspend_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker String,
                        trade_date DateTime,
                        suspend_timing Nullable(String),
                        suspend_type String,
                        PRIMARY KEY trade_date
                    )
                    ENGINE = MergeTree
                    ORDER BY trade_date;
                    """)
    
    def create_stock_basic_table(self, table_name):
        self._query(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}
                    (
                        ticker String,
                        name String,
                        area String,
                        industry String,
                        fullname String,
                        enname String,
                        market String,
                        exchange String,
                        list_date DateTime,
                        PRIMARY KEY ticker
                    )
                    ENGINE = ReplacingMergeTree
                    ORDER BY list_date;
                    """)
