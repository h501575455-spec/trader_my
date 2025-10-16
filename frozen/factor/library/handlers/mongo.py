import pandas as pd
import datetime
from typing import Optional, Dict, List, Any
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne, errors
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import FactorHandler
from ...expression.base import Factor
from ....data.database import build_config, DatabaseTypes
from ....utils.log import GL


class MongoFactorHandler(FactorHandler):
    """MongoDB implementation of FactorHandler"""
    
    def __init__(self):
        self.config = build_config(DatabaseTypes.MONGODB.value)
        self.conn = None
        self.db = None
        self._logger = GL.get_logger(__name__)
        self._lifecycle_collection_name = "factor_lifecycle"
    
    def init_db(self) -> None:
        """Initialize MongoDB connection"""
        client = MongoClient(host=self.config.host, port=self.config.port)
        self.conn = client.FactorBase
        self.db = self.conn
    
    def factor2db(self, factor: Factor, factor_name: str, table_name: str, 
                  parallel: bool = False) -> None:
        """Store factor data to MongoDB"""
        if self.conn is None:
            self.init_db()
        
        # Transform factor data
        factor_data = self._transform_factor_data(factor, factor_name)
        
        # Check if collection exists
        if not self._check_table_exists(table_name):
            self._create_collection_and_insert_data(table_name, factor_data)
        else:
            self._update_existing_data(table_name, factor_data, factor_name, parallel)
    
    def read_factor(self, table_name: str, factor_name: str, 
                   start_date: str = None, end_date: str = None) -> Factor:
        """Read factor data from MongoDB"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn[table_name]
        query = {}
        if start_date or end_date:
            query["trade_date"] = {}
            if start_date:
                query["trade_date"]["$gte"] = pd.to_datetime(start_date)
            if end_date:
                query["trade_date"]["$lte"] = pd.to_datetime(end_date)
        
        cursor = collection.find(query, {"ticker": 1, "trade_date": 1, factor_name: 1})
        data = pd.DataFrame(list(cursor))
        
        if data.empty:
            return Factor(pd.DataFrame())
        
        del data["_id"]
        data.set_index(["ticker", "trade_date"], inplace=True)
        data = data.swaplevel().unstack()
        data.columns = data.columns.droplevel(level=0)
        data.sort_index(inplace=True)
        
        return Factor(data)
    
    def check_factor_exists(self, table_name: str, factor_name: str) -> bool:
        """Check if factor exists in MongoDB collection"""
        if self.conn is None:
            self.init_db()
        
        if table_name not in self.conn.list_collection_names():
            return False
        
        collection = self.conn[table_name]
        return bool(list(collection.find({factor_name: {"$exists": True}}).limit(1)))
    
    def get_factor_date_range(self, table_name: str, factor_name: str) -> tuple:
        """Get factor data date range from MongoDB"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn[table_name]
        pipeline = [
            {"$match": {factor_name: {"$exists": True}}},
            {"$group": {
                "_id": None,
                "min_date": {"$min": "$trade_date"},
                "max_date": {"$max": "$trade_date"}
            }}
        ]
        
        result = list(collection.aggregate(pipeline))
        if result:
            return result[0]["min_date"], result[0]["max_date"]
        return None, None
    
    def delete_factor(self, table_name: str, factor_name: str) -> None:
        """Delete factor from MongoDB collection"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn[table_name]
        collection.update_many({}, {"$unset": {factor_name: ""}})
    
    def _transform_factor_data(self, factor: Factor, factor_name: str):
        """Transform factor data for MongoDB storage"""
        factor_data = factor.data.stack()
        factor_data.name = factor_name
        factor_data.index.names = ["trade_date", "ticker"]
        factor_data = factor_data.swaplevel().sort_index().reset_index()
        return factor_data.to_dict(orient="records")
    
    def _check_table_exists(self, table_name: str) -> bool:
        """Check if MongoDB collection exists and is not empty"""
        collection = self.conn[table_name]
        return bool(list(collection.find().limit(1)))
    
    def _create_collection_and_insert_data(self, table_name: str, factor_data: list):
        """Create new MongoDB collection and insert initial data"""
        collection = self.conn[table_name]
        collection.create_index([("ticker", 1), ("trade_date", 1)], unique=True)
        collection.insert_many(factor_data, ordered=False)
        print(f"Collection '{table_name}' created and initial factor data inserted.")
    
    def _update_existing_data(self, table_name: str, factor_data: list, 
                             factor_name: str, parallel: bool = False):
        """Update existing MongoDB collection with new factor data"""
        collection = self.conn[table_name]
        
        updates = [
            UpdateOne(
                {"ticker": item["ticker"], "trade_date": item["trade_date"]},
                {"$set": {factor_name: item[factor_name]}},
                upsert=True
            )
            for item in factor_data
        ]
        
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                chunks = [updates[i:i + 1000] for i in range(0, len(updates), 1000)]
                future_to_chunk = {executor.submit(collection.bulk_write, chunk): chunk 
                                 for chunk in chunks}
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        future.result()
                        print(f"Processed chunk of {len(chunk)} updates.")
                    except Exception as e:
                        print(f"Error processing chunk: {e}")
        else:
            for i in tqdm(range(0, len(updates), 1000)):
                batch = updates[i:i + 1000]
                try:
                    result = collection.bulk_write(batch, ordered=False)
                    print(f"Batch update result: {result.bulk_api_result}")
                except errors.BulkWriteError as e:
                    print(f"Bulk write error: {e.details}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.conn:
            self.conn.client.close()
    
    # Factor metadata management methods
    def save_factor_metadata(self, metadata) -> None:
        """Save factor metadata to MongoDB"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn["factor_metadata"]
        
        # Convert metadata to dict
        metadata_dict = {
            "uid": metadata.uid,
            "name": metadata.name,
            "description": metadata.description,
            "dependencies": metadata.dependencies,
            "category": metadata.category,
            "tags": metadata.tags,
            "author": getattr(metadata, "author", ""),
            "lifecycle_status": getattr(metadata, "lifecycle_status", "development").value if hasattr(getattr(metadata, "lifecycle_status", "development"), 'value') else str(getattr(metadata, "lifecycle_status", "development")),
            "version": str(getattr(metadata, "version", "1.0.0")),
            "created_time": metadata.created_time,
            "updated_time": datetime.datetime.now()
        }
        
        # Upsert metadata by UID
        collection.replace_one(
            {"uid": metadata.uid},
            metadata_dict,
            upsert=True
        )
        
        # Also create/update index by name for lookups
        collection.create_index("name", unique=True, background=True)
        collection.create_index("uid", unique=True, background=True)
    
    def load_factor_metadata(self, factor_name: str):
        """Load factor metadata from MongoDB"""
        if self.conn is None:
            self.init_db()

        collection = self.conn["factor_metadata"]
        doc = collection.find_one({"name": factor_name})
        
        if doc:
            # Import here to avoid circular import
            from ..inventory import FactorMetadata
            from ..lifecycle import FactorLifecycleStatus, FactorVersion
            
            # Convert lifecycle_status string to enum
            lifecycle_status_value = doc.get("lifecycle_status", "development")
            try:
                lifecycle_status = FactorLifecycleStatus(lifecycle_status_value)
            except (ValueError, TypeError):
                lifecycle_status = FactorLifecycleStatus.DEVELOPMENT
            
            # Convert version string to FactorVersion object
            version_str = doc.get("version", "1.0.0")
            try:
                version = FactorVersion.from_string(version_str) if hasattr(FactorVersion, 'from_string') else FactorVersion()
            except:
                version = FactorVersion()
            
            return FactorMetadata(
                uid=doc.get("uid", ""),
                name=doc["name"],
                description=doc.get("description", ""),
                dependencies=doc.get("dependencies", []),
                category=doc.get("category", "default"),
                tags=doc.get("tags", []),
                author=doc.get("author", ""),
                lifecycle_status=lifecycle_status,
                version=version,
                created_time=doc.get("created_time"),
                updated_time=doc.get("updated_time")
            )
        return None
    
    def load_all_factor_metadata(self) -> Dict[str, any]:
        """Load all factor metadata from MongoDB"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn["factor_metadata"]
        docs = collection.find({})
        
        result = {}
        for doc in docs:
            # Import here to avoid circular import
            from ..inventory import FactorMetadata
            from ..lifecycle import FactorLifecycleStatus, FactorVersion
            
            # Convert lifecycle_status string to enum
            lifecycle_status_value = doc.get("lifecycle_status", "development")
            try:
                lifecycle_status = FactorLifecycleStatus(lifecycle_status_value)
            except (ValueError, TypeError):
                lifecycle_status = FactorLifecycleStatus.DEVELOPMENT
            
            # Convert version string to FactorVersion object
            version_str = doc.get("version", "1.0.0")
            try:
                version = FactorVersion.from_string(version_str) if hasattr(FactorVersion, 'from_string') else FactorVersion()
            except:
                version = FactorVersion()
            
            metadata = FactorMetadata(
                uid=doc.get("uid", ""),
                name=doc["name"],
                description=doc.get("description", ""),
                dependencies=doc.get("dependencies", []),
                category=doc.get("category", "default"),
                tags=doc.get("tags", []),
                namespace=doc.get("namespace", "default"),
                author=doc.get("author", ""),
                lifecycle_status=lifecycle_status,
                version=version,
                created_time=doc.get("created_time"),
                updated_time=doc.get("updated_time")
            )
            # Use UID as key instead of name for the new format
            result[doc.get("uid", doc["name"])] = metadata
        
        return result
    
    def delete_factor_metadata(self, factor_name: str) -> None:
        """Delete factor metadata from MongoDB"""
        if self.conn is None:
            self.init_db()
        
        collection = self.conn["factor_metadata"]
        collection.delete_one({"name": factor_name})
    
    def check_metadata_table_exists(self) -> bool:
        """Check if metadata collection exists in MongoDB"""
        if self.conn is None:
            self.init_db()
        
        return "factor_metadata" in self.conn.list_collection_names()
    
    def close(self) -> None:
        """Close MongoDB connection"""
        if self.conn is not None:
            self.conn.client.close()
            self.conn = None
            self.db = None

    # Lifecycle management methods
    def _ensure_lifecycle_collections(self) -> None:
        """Ensure lifecycle metadata collection exists with proper indices"""
        if self._lifecycle_collection_name not in self.db.list_collection_names():
            self.db.create_collection(self._lifecycle_collection_name)
            # Create indices for efficient querying
            lifecycle_collection = self.db[self._lifecycle_collection_name]
            lifecycle_collection.create_index("factor_name")
            lifecycle_collection.create_index("status")
            lifecycle_collection.create_index("created_at")
            self._logger.info("Created lifecycle metadata collection with indices")
    
    def save_lifecycle_metadata(self, factor_name: str, lifecycle_meta) -> bool:
        """Save lifecycle metadata to MongoDB"""
        try:
            # Ensure lifecycle collections exist before saving
            self._ensure_lifecycle_collections()
            
            collection = self.db[self._lifecycle_collection_name]
            data = self._serialize_lifecycle_for_db(lifecycle_meta)
            data["factor_name"] = factor_name
            
            # Upsert the document
            result = collection.replace_one(
                {"factor_name": factor_name},
                data,
                upsert=True
            )
            
            self._logger.info(f"Saved lifecycle metadata for factor '{factor_name}'")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save lifecycle metadata for '{factor_name}': {e}")
            return False
    
    def load_lifecycle_metadata(self, factor_name: str):
        """Load lifecycle metadata from MongoDB"""
        try:
            collection = self.db[self._lifecycle_collection_name]
            data = collection.find_one({"factor_name": factor_name})
            
            if data:
                return self._deserialize_lifecycle_from_db(data)
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to load lifecycle metadata for '{factor_name}': {e}")
            return None
    
    def load_all_lifecycle_metadata(self) -> Dict[str, Any]:
        """Load all lifecycle metadata from MongoDB"""
        try:
            collection = self.db[self._lifecycle_collection_name]
            result = {}
            
            for data in collection.find():
                factor_name = data["factor_name"]
                lifecycle_meta = self._deserialize_lifecycle_from_db(data)
                result[factor_name] = lifecycle_meta
            
            return result
            
        except Exception as e:
            self._logger.error(f"Failed to load all lifecycle metadata: {e}")
            return {}
    
    def delete_lifecycle_metadata(self, factor_name: str) -> bool:
        """Delete lifecycle metadata from MongoDB"""
        try:
            collection = self.db[self._lifecycle_collection_name]
            result = collection.delete_one({"factor_name": factor_name})
            
            success = result.deleted_count > 0
            if success:
                self._logger.info(f"Deleted lifecycle metadata for factor '{factor_name}'")
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to delete lifecycle metadata for '{factor_name}': {e}")
            return False
    
    def query_factors_by_status(self, status) -> List[str]:
        """Query factors by lifecycle status"""
        try:
            collection = self.db[self._lifecycle_collection_name]
            cursor = collection.find({"status": status.value}, {"factor_name": 1})
            return [doc["factor_name"] for doc in cursor]
            
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