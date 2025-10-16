from enum import Enum

class DatabaseTypes(Enum):
    MONGODB = "mongodb"
    DUCKDB = "duckdb"
    CHDB = "chdb"

def get_data_config():
    """Lazy loading database config to avoid circular import"""
    from ..config import data_config
    return data_config["database"]

class DatabaseConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host=None, port=None, username=None, password=None, data_path=None, factor_path=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.data_path = data_path
        self.factor_path = factor_path
    
    def connect(self):
        return f"Connecting to database at {self.host}:{self.port} with user {self.username} and path {self.data_path}."


class ConfigBuilder:
    def __init__(self, host=None, port=None, username=None, password=None, **kwargs):
        self._config = {
            "host": host,
            "port": port,
            "username": username,
            "password": password
        }
    
    def set_data_file_path(self, file_path):
        self._config["data_path"] = file_path
        return self
    
    def set_factor_file_path(self, file_path):
        self._config["factor_path"] = file_path
        return self
    
    def build(self):
        return DatabaseConfig(**self._config)


def build_config(db_type):

    database_cfg = get_data_config()
    
    builder = ConfigBuilder(**database_cfg[db_type])
    if db_type == "mongodb":
        pass
    elif db_type == "duckdb":
        builder.set_data_file_path(database_cfg[db_type]["data_path"])\
               .set_factor_file_path(database_cfg[db_type]["factor_path"])
    elif db_type == "chdb":
        builder.set_data_file_path(database_cfg[db_type]["data_path"])\
               .set_factor_file_path(database_cfg[db_type]["factor_path"])
    else:
        raise NotImplementedError(f"Database {db_type} not implemented yet.")
    return builder.build()