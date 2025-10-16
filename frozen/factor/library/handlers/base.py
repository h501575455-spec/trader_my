import abc
import json
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from ...expression.base import Factor


class FactorHandler(abc.ABC):
    """Abstract base class for factor database operations"""
    
    @abc.abstractmethod
    def init_db(self) -> None:
        """Initialize database connection"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def factor2db(self, factor: Factor, factor_name: str, table_name: str, **kwargs) -> None:
        """Store factor data to database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def read_factor(self, table_name: str, factor_name: str, 
                   start_date: str = None, end_date: str = None) -> Factor:
        """Read factor data from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def check_factor_exists(self, table_name: str, factor_name: str) -> bool:
        """Check if factor exists in database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_factor_date_range(self, table_name: str, factor_name: str) -> tuple:
        """Get factor data date range"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete_factor(self, table_name: str, factor_name: str) -> None:
        """Delete factor from database"""
        raise NotImplementedError
    
    # Factor metadata management methods
    @abc.abstractmethod
    def save_factor_metadata(self, metadata) -> None:
        """Save factor metadata to database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_factor_metadata(self, factor_name: str):
        """Load factor metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_all_factor_metadata(self) -> Dict[str, any]:
        """Load all factor metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete_factor_metadata(self, factor_name: str) -> None:
        """Delete factor metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def check_metadata_table_exists(self) -> bool:
        """Check if metadata table/collection exists"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self) -> None:
        """Close database connection and cleanup resources"""
        raise NotImplementedError

    # Lifecycle management methods
    @abc.abstractmethod
    def save_lifecycle_metadata(self, factor_name: str, lifecycle_meta) -> bool:
        """Save lifecycle metadata to database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_lifecycle_metadata(self, factor_name: str):
        """Load lifecycle metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_all_lifecycle_metadata(self) -> Dict[str, Any]:
        """Load all lifecycle metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def delete_lifecycle_metadata(self, factor_name: str) -> bool:
        """Delete lifecycle metadata from database"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def query_factors_by_status(self, status) -> List[str]:
        """Query factors by lifecycle status"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_lifecycle_history(self, factor_name: str, limit: int = 100) -> List[Dict]:
        """Get lifecycle transition history for a factor"""
        raise NotImplementedError
    
    def _serialize_lifecycle_for_db(self, lifecycle_meta) -> Dict:
        """Serialize lifecycle metadata for database storage"""
        from ..lifecycle import LifecycleMetadata
        
        if not isinstance(lifecycle_meta, LifecycleMetadata):
            raise TypeError("Expected LifecycleMetadata object")
        
        data = {
            "status": lifecycle_meta.status.value,
            "version": {
                "major": lifecycle_meta.version.major,
                "minor": lifecycle_meta.version.minor,
                "patch": lifecycle_meta.version.patch
            },
            "status_history": [],
            "monitoring_enabled": lifecycle_meta.monitoring_enabled,
            "alert_contacts": lifecycle_meta.alert_contacts,
            "version_history": lifecycle_meta.version_history,
            "parent_version": lifecycle_meta.parent_version,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Serialize status history
        for record in lifecycle_meta.status_history:
            data["status_history"].append({
                "from_status": record.from_status.value,
                "to_status": record.to_status.value,
                "timestamp": record.timestamp.isoformat(),
                "operator": record.operator,
                "reason": record.reason,
                "metadata": record.metadata
            })
        
        return data
    
    def _deserialize_lifecycle_from_db(self, data: Dict):
        """Deserialize lifecycle metadata from database"""
        from ..lifecycle import (
            LifecycleMetadata, StatusTransitionRecord, 
            FactorVersion, FactorLifecycleStatus
        )
        
        # Reconstruct version
        version_data = data.get("version", {})
        version = FactorVersion(
            major=version_data.get("major", 1),
            minor=version_data.get("minor", 0),
            patch=version_data.get("patch", 0)
        )
        
        # Reconstruct status history
        status_history = []
        for record_data in data.get("status_history", []):
            status_history.append(StatusTransitionRecord(
                from_status=FactorLifecycleStatus(record_data["from_status"]),
                to_status=FactorLifecycleStatus(record_data["to_status"]),
                timestamp=datetime.datetime.fromisoformat(record_data["timestamp"]),
                operator=record_data["operator"],
                reason=record_data["reason"],
                metadata=record_data.get("metadata", {})
            ))
        
        return LifecycleMetadata(
            status=FactorLifecycleStatus(data["status"]),
            version=version,
            status_history=status_history,
            monitoring_enabled=data.get("monitoring_enabled", False),
            alert_contacts=data.get("alert_contacts", []),
            version_history=data.get("version_history", []),
            parent_version=data.get("parent_version")
        )