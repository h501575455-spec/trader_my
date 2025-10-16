from .mongo.handler import MongoDBHandler
from .duck.handler import DuckDBHandler
from .ch.handler import ChdbHandler

from . import DatabaseTypes


class DatabaseFactory:
    
    @staticmethod
    def create_database_connection(database_type: DatabaseTypes):
        if database_type == DatabaseTypes.MONGODB:
            return MongoDBHandler()
        elif database_type == DatabaseTypes.DUCKDB:
            return DuckDBHandler()
        elif database_type == DatabaseTypes.CHDB:
            return ChdbHandler()
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    @staticmethod
    def create_data_loader(database_type: DatabaseTypes):
        if database_type == DatabaseTypes.DUCKDB:
            from .duck.loader import DuckDBLoader
            return DuckDBLoader()
        elif database_type == DatabaseTypes.MONGODB:
            from .mongo.loader import MongoDBLoader
            return MongoDBLoader()
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
