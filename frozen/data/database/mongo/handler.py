import datetime
import pandas as pd
from typing import Union, Dict
from pymongo import MongoClient, errors

from ..base import DatabaseHandler
from .. import build_config, DatabaseTypes

class MongoDBHandler(DatabaseHandler):
    """MongoDB database handler"""
    
    def __init__(self):
        self.config = build_config(DatabaseTypes.MONGODB.value)
    
    def init_db(self):
        try:
            self.config.connect()
            client = MongoClient(
                host=self.config.host, 
                port=self.config.port,
                serverSelectionTimeoutMS=5000
            )
            client.server_info()
            self.db = client.FrozenTest
            print(f"Successfully connected to MongoDB at {self.config.host}:{self.config.port}")
        except errors.ServerSelectionTimeoutError as e:
            print(f"ERROR: Cannot connect to MongoDB server: {e}")
            print("Please check if MongoDB is running and accessible")
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error initializing MongoDB: {e}")
            raise

    def _check_table_exists(self, table_name=""):
        res = table_name in self.db.list_collection_names()
        return res
    
    def _check_table_empty(self, table_name):        
        query_str = {
            "collection": f"{table_name}",
            "action": "count_documents"
        }
        result = self._query(query_str)
        res = result == 0
        return res
    
    def _check_data_exists(self, table_name, data_str):
        if "." in data_str:
            query_str = {
                "collection": f"{table_name}",
                "action": "find",
                "filter": {"ticker": f"{data_str}"}
            }
        else:
            formatted_data = datetime.datetime.strptime(data_str, "%Y-%m-%d")
            query_str = {
                "collection": f"{table_name}",
                "action": "find",
                "filter": {"trade_date": f"{formatted_data}"}
            }
        result = self._query(query_str)
        return result
    
    def _get_table_ticker(self, table_name):
        query_str = {
            "collection": f"{table_name}",
            "action": "distinct",
            "field": "ticker"
        }
        res = self._query(query_str)
        return res
    
    def _get_table_date(self, table_name, latest=False) -> Union[pd.DataFrame, pd.Timestamp]:
        date_col = "ann_date" if table_name == "stock_dividend" else "trade_date"
        query_str = {
            "collection": f"{table_name}",
            "action": "aggregate",
            "aggregate": {
                "$group": {
                    "_id": "$ticker", 
                    "max_date": {"$max": f"${date_col}"}}
            }
        }
        table_date = self._query(query_str, fmt="dataframe")
        if latest:
            table_date = table_date["max_date"].max()
        return table_date
    
    def _get_ticker_date(self, table_date, ticker, shift=0) -> str:
        ticker_date = table_date[table_date["_id"]==ticker]["max_date"].iloc[0]
        if isinstance(ticker_date, pd.Timestamp):
            ticker_date = (ticker_date + datetime.timedelta(shift)).strftime("%Y%m%d")
        elif isinstance(ticker_date, str):
            ticker_date = (datetime.datetime.strptime(ticker_date, "%Y%m%d") + datetime.timedelta(shift)).strftime("%Y%m%d")
        else:
            raise ValueError(f"Invalid ticker date type: {type(ticker_date)}")
        return ticker_date
    
    def _get_latest_ticker_date(self, table_name, ticker):
        table_date = self._get_table_date(table_name, latest=False)
        start_date = self._get_ticker_date(table_date, ticker, shift=1)
        return start_date

    def _get_table_earliest_date(self, table_name) -> pd.DataFrame:
        """Get the earliest date for each ticker in the table"""
        date_col = "ann_date" if table_name == "stock_dividend" else "trade_date"
        query_str = {
            "collection": f"{table_name}",
            "action": "aggregate",
            "aggregate": {
                "$group": {
                    "_id": "$ticker",
                    "min_date": {"$min": f"${date_col}"}}
            }
        }
        table_earliest_date = self._query(query_str, fmt="dataframe")
        return table_earliest_date

    def _get_earliest_ticker_date(self, table_name, ticker):
        """Get the earliest date for a specific ticker in the table"""
        table_earliest_date = self._get_table_earliest_date(table_name)
        if ticker not in table_earliest_date["_id"].values:
            return None
        earliest_date = table_earliest_date[table_earliest_date["_id"]==ticker]["min_date"].iloc[0]
        if isinstance(earliest_date, pd.Timestamp):
            earliest_date = earliest_date.strftime("%Y%m%d")
        return earliest_date

    def _insert_df_to_table(self, df, table_name):
        docs = df.to_dict(orient="records")
        query_str = {
            "collection": f"{table_name}",
            "action": "insert_many",
            "documents": docs
        }
        self._query(query_str)

    def _delete_table(self, table_name):
        query_str = {
        "collection": f"{table_name}",
        "action": "drop_collection"
        }
        self._query(query_str)

    def _clear_table(self, table_name):
        query_str = {
            "collection": f"{table_name}",
            "action": "delete_many"
        }
        self._query(query_str)
    
    def _query(self, query_str: Union[str, Dict], fmt=None):
        """
        Execute query and return results in specified format.
        
        Args:
            query_str: Mongo query dictionary
            fmt: Return format
                - None (default)
                - "dataframe"
        
        Returns:
            Query results in specified format or None for non-SELECT queries
        """
        if fmt is None:
            fmt = "dataframe"
        
        # MongoDB query handling
        collection_name = query_str["collection"]
        action = query_str.get("action", "find")
        
        collection = self.db[collection_name]
        
        try:
            if action == "find":
                # Regular find query
                filter_query = query_str.get("filter", {})
                projection = query_str.get("projection", {"_id": 0})
                res = list(collection.find(filter_query, projection))
            
            elif action == "create_index":
                # Create index action
                index_fields = query_str["index_fields"]
                unique = query_str.get("unique", False)
                res = collection.create_index(index_fields, unique=unique)

            elif action == "insert_many":
                # Insert many action
                documents = query_str["documents"]
                res = collection.insert_many(documents)
            
            elif action == "aggregate":
                # Aggregate action
                agg_query = [query_str.get("aggregate", {})]
                res = list(collection.aggregate(agg_query))
            
            elif action == "drop_collection":
                # Drop collection action
                res = collection.drop()
            
            elif action == "delete_many":
                # Delete many action
                filter_query = query_str.get("filter", {})
                res = collection.delete_many(filter_query)
            
            elif action == "count_documents":
                # Count documents action
                filter_query = query_str.get("filter", {})
                res = collection.count_documents(filter_query)
            
            elif action == "distinct":
                # Distinct action
                field = query_str["field"]
                res = collection.distinct(field)

            if fmt == "dataframe" and res:
                try:
                    res = pd.DataFrame(res)
                except Exception as e:
                    res = None
        
            return res
        
        except errors.PyMongoError as e:
            print(f"Error executing MongoDB operation: {e}")
            return pd.DataFrame() if fmt == "dataframe" else []

    def create_volume_price_table(self, table_name):
        self.db.create_collection(table_name)
        query_str = {
            "collection": f"{table_name}",
            "action": "create_index",
            "index_fields": [("ticker", 1), ("trade_date", 1)],
            "unique": True
        }
        self._query(query_str)
    
    def create_stock_limit_table(self, table_name):
        self.db.create_collection(table_name)
        query_str = {
            "collection": f"{table_name}",
            "action": "create_index",
            "index_fields": [("ticker", 1), ("trade_date", 1)],
            "unique": True
        }
        self._query(query_str)
    
    def create_stock_fundamental_table(self, table_name):
        self.db.create_collection(table_name)
        query_str = {
            "collection": f"{table_name}",
            "action": "create_index",
            "index_fields": [("ticker", 1), ("trade_date", 1)],
            "unique": True
        }
        self._query(query_str)
    
    def create_stock_dividend_table(self, table_name):
        self.db.create_collection(table_name)
        query_str = {
            "collection": f"{table_name}",
            "action": "create_index",
            "index_fields": [("ticker", 1), ("end_date", 1), ("ann_date", 1), ("div_proc", 1)],
            "unique": True
        }
        self._query(query_str)
    
    def create_stock_suspend_table(self, table_name):
        self.db.create_collection(table_name)
    
    def create_stock_basic_table(self, table_name):
        self.db.create_collection(table_name)
        query_str = {
            "collection": f"{table_name}",
            "action": "create_index",
            "index_fields": [("ticker", 1)],
            "unique": True
        }
        self._query(query_str)

