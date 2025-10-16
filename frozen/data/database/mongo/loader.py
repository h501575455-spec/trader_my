import pandas as pd
from ..factory import DatabaseFactory
from .. import DatabaseTypes


class MongoDBLoader:
    
    def __init__(self):
        self.handler = DatabaseFactory.create_database_connection(DatabaseTypes.MONGODB)
        self.handler.init_db()

    def load_time_series_data(self, table_name, universe, start_date, end_date):
        query_str = {
            "collection": f"{table_name}",
            "filter": {"ticker": {"$in": list(universe)}, "trade_date": {"$gte": start_date, "$lte": end_date}},
        }
        data = self.handler._query(query_str)
        return data
    
    def load_basic_data(self, table_name):
        query_str = {
            "collection": f"{table_name}",
            "filter": {},
        }
        data = self.handler._query(query_str)
        return data
    
    def load_dividend_data(self, table_name, universe):
        query_str = {
            "collection": f"{table_name}",
            "filter": {"ticker": {"$in": list(universe)}}
        }
        data = self.handler._query(query_str)
        return data
    
    def transform_dividend_data(self, raw_data):
        data = {
            key: (
                group
                # Step 1: screen out the data with `div_proc` as `实施`
                .query("div_proc=='实施'")
                .drop(["ticker"], axis=1)
                .sort_values(by="ex_date")
                .reset_index(drop=True)
                # Step 2: sum the data by `ex_date`
                .groupby("ex_date", as_index=False)
                .agg({
                    **{col: "sum" for col in ["stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"]},  # take the sum
                    # **{col: "first" for col in ["stk_div", "stk_bo_rate", "stk_co_rate", "cash_div", "cash_div_tax"]}  # keep first value
                })
            )
            for key, group in raw_data.groupby("ticker")
        }
        return data
    
    def load_suspend_data(self, table_name, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        query_str = {
            "collection": f"{table_name}",
            "filter": {"trade_date": {"$gte": start_date, "$lte": end_date}}
        }
        data = self.handler._query(query_str)
        return data
    
    def transform_suspend_data(self, raw_data):
        data = raw_data.sort_values(by=["trade_date", "ticker"], ascending=True)
        data.set_index("trade_date", inplace=True)
        return data