import datetime
import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path

from ...utils.log import GL


class FactorLifecycleStatus(Enum):
    """Factor lifecycle status enumeration"""
    # Development phase
    DEVELOPMENT = "development"      # In development - Writing and debugging
    
    # Review phase  
    REVIEW = "review"                # Under review - Awaiting manual review
    
    # Production phase
    ACTIVE = "active"                # Active - In production use
    
    # Termination phase
    DEPRECATED = "deprecated"        # Deprecated - Planned for retirement
    ARCHIVED = "archived"            # Archived - Historical preservation state


@dataclass
class FactorVersion:
    """Factor version information"""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other):
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> "FactorVersion":
        """Create version object from string"""
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    
    def increment_patch(self) -> "FactorVersion":
        """Increment patch version"""
        return FactorVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> "FactorVersion":
        """Increment minor version"""
        return FactorVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> "FactorVersion":
        """Increment major version"""
        return FactorVersion(self.major + 1, 0, 0)


@dataclass
class StatusTransitionRecord:
    """Status transition record"""
    from_status: FactorLifecycleStatus
    to_status: FactorLifecycleStatus
    timestamp: datetime.datetime
    operator: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecycleMetadata:
    """Factor lifecycle metadata"""
    status: FactorLifecycleStatus = FactorLifecycleStatus.DEVELOPMENT
    version: FactorVersion = field(default_factory=FactorVersion)
    status_history: List[StatusTransitionRecord] = field(default_factory=list)
    
    # Monitoring configuration
    monitoring_enabled: bool = False
    alert_contacts: List[str] = field(default_factory=list)
    
    # Version control
    version_history: List[Dict] = field(default_factory=list)
    parent_version: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.status_history:
            self.status_history = [
                StatusTransitionRecord(
                    from_status=FactorLifecycleStatus.DEVELOPMENT,
                    to_status=self.status,
                    timestamp=datetime.datetime.now(),
                    operator="system",
                    reason="Initial creation"
                )
            ]


class FactorLifecycleTransition:
    """Factor lifecycle transition rules"""
    
    # Define status order for transition validation
    STATUS_ORDER = {
        FactorLifecycleStatus.DEVELOPMENT: 1,
        FactorLifecycleStatus.REVIEW: 2,
        FactorLifecycleStatus.ACTIVE: 3,
        FactorLifecycleStatus.DEPRECATED: 4,
        FactorLifecycleStatus.ARCHIVED: 5
    }
    
    # Define allowed status transitions (forward only)
    ALLOWED_TRANSITIONS = {
        FactorLifecycleStatus.DEVELOPMENT: [
            FactorLifecycleStatus.REVIEW,
            FactorLifecycleStatus.ACTIVE,
            FactorLifecycleStatus.DEPRECATED,
            FactorLifecycleStatus.ARCHIVED
        ],
        FactorLifecycleStatus.REVIEW: [
            FactorLifecycleStatus.ACTIVE,
            FactorLifecycleStatus.DEPRECATED,
            FactorLifecycleStatus.ARCHIVED
        ],
        FactorLifecycleStatus.ACTIVE: [
            FactorLifecycleStatus.DEPRECATED,
            FactorLifecycleStatus.ARCHIVED
        ],
        FactorLifecycleStatus.DEPRECATED: [
            FactorLifecycleStatus.ARCHIVED
        ],
        FactorLifecycleStatus.ARCHIVED: []  # Terminal state
    }
    
    @classmethod
    def is_transition_allowed(cls, from_status: FactorLifecycleStatus, 
                            to_status: FactorLifecycleStatus) -> bool:
        """Check if status transition is allowed (forward only)"""
        # Allow same status (no change)
        if from_status == to_status:
            return True
            
        # Check if transition is in allowed list
        allowed = cls.ALLOWED_TRANSITIONS.get(from_status, [])
        return to_status in allowed
    
    @classmethod
    def get_allowed_transitions(cls, from_status: FactorLifecycleStatus) -> List[FactorLifecycleStatus]:
        """Get allowed transition states"""
        return cls.ALLOWED_TRANSITIONS.get(from_status, [])


class FactorVersionManager:
    """Factor version manager"""
    
    def __init__(self, base_path: str = "./factor_versions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._logger = GL.get_logger(__name__)
    
    def create_version(self, factor_name: str, version: FactorVersion, 
                      source_code: str = "", metadata: Dict = None) -> bool:
        """Create new version"""
        version_dir = self.base_path / factor_name / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_info = {
            "version": str(version),
            "created_at": datetime.datetime.now().isoformat(),
            "source_code": source_code,
            "metadata": metadata or {}
        }
        
        # Save version information
        version_file = version_dir / "version.json"
        with open(version_file, "w", encoding="utf-8") as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        # Save source code
        if source_code:
            code_file = version_dir / "factor.py"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(source_code)
        
        self._logger.info(f"Created version {version} for factor {factor_name}")
        return True
    
    def get_versions(self, factor_name: str) -> List[FactorVersion]:
        """Get all versions of the factor"""
        factor_dir = self.base_path / factor_name
        if not factor_dir.exists():
            return []
        
        versions = []
        for version_dir in factor_dir.iterdir():
            if version_dir.is_dir():
                try:
                    version = FactorVersion.from_string(version_dir.name)
                    versions.append(version)
                except ValueError:
                    continue
        
        return sorted(versions)
    
    def get_latest_version(self, factor_name: str) -> Optional[FactorVersion]:
        """Get latest version"""
        versions = self.get_versions(factor_name)
        return max(versions) if versions else None
    
    def get_version_info(self, factor_name: str, version: FactorVersion) -> Dict:
        """Get version information"""
        version_file = self.base_path / factor_name / str(version) / "version.json"
        if not version_file.exists():
            return {}
        
        with open(version_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def compare_versions(self, factor_name: str, version1: FactorVersion, 
                        version2: FactorVersion) -> Dict:
        """Compare two versions"""
        info1 = self.get_version_info(factor_name, version1)
        info2 = self.get_version_info(factor_name, version2)
        
        return {
            "version1": {
                "version": str(version1),
                "created_at": info1.get("created_at"),
                "metadata": info1.get("metadata", {})
            },
            "version2": {
                "version": str(version2),
                "created_at": info2.get("created_at"),
                "metadata": info2.get("metadata", {})
            },
            "diff": self._calculate_diff(info1, info2)
        }
    
    def _calculate_diff(self, info1: Dict, info2: Dict) -> Dict:
        """Calculate version differences"""
        diff = {
            "source_code_changed": info1.get("source_code") != info2.get("source_code"),
            "metadata_changes": {}
        }
        
        meta1 = info1.get("metadata", {})
        meta2 = info2.get("metadata", {})
        
        all_keys = set(meta1.keys()) | set(meta2.keys())
        for key in all_keys:
            if key not in meta1:
                diff["metadata_changes"][key] = {"type": "added", "new_value": meta2[key]}
            elif key not in meta2:
                diff["metadata_changes"][key] = {"type": "removed", "old_value": meta1[key]}
            elif meta1[key] != meta2[key]:
                diff["metadata_changes"][key] = {
                    "type": "modified", 
                    "old_value": meta1[key], 
                    "new_value": meta2[key]
                }
        
        return diff


class FactorLifecycleManager:
    """Factor lifecycle manager"""
    
    def __init__(self, factor_registry=None, version_manager: FactorVersionManager = None,
                 performance_validator: Callable = None):
        self.factor_registry = factor_registry
        self.version_manager = version_manager or FactorVersionManager()
        self.performance_validator = performance_validator
        self.transition_rules = FactorLifecycleTransition()
        self._logger = GL.get_logger(__name__)
        
        # Store lifecycle metadata
        self._lifecycle_metadata: Dict[str, LifecycleMetadata] = {}
    
    def initialize_factor_lifecycle(self, factor_name: str, 
                                   initial_status: FactorLifecycleStatus = FactorLifecycleStatus.DEVELOPMENT,
                                   operator: str = "system") -> LifecycleMetadata:
        """Initialize factor lifecycle"""
        if factor_name in self._lifecycle_metadata:
            return self._lifecycle_metadata[factor_name]
        
        lifecycle_meta = LifecycleMetadata(status=initial_status)
        lifecycle_meta.status_history = [
            StatusTransitionRecord(
                from_status=initial_status,
                to_status=initial_status,
                timestamp=datetime.datetime.now(),
                operator=operator,
                reason="Factor lifecycle initialization"
            )
        ]
        
        self._lifecycle_metadata[factor_name] = lifecycle_meta
        self._logger.info(f"Initialized lifecycle for factor '{factor_name}' with status {initial_status.value}")
        
        return lifecycle_meta
    
    def transition_status(self, factor_name: str, 
                         new_status: FactorLifecycleStatus,
                         reason: str = "",
                         operator: str = "system",
                         metadata: Dict = None) -> bool:
        """Status transition"""
        if factor_name not in self._lifecycle_metadata:
            # Try to load from database first
            if self.factor_registry:
                existing_lifecycle = self.factor_registry._handler.load_lifecycle_metadata(factor_name)
                if existing_lifecycle:
                    self._lifecycle_metadata[factor_name] = existing_lifecycle
                else:
                    self.initialize_factor_lifecycle(factor_name)
            else:
                self.initialize_factor_lifecycle(factor_name)
        
        lifecycle_meta = self._lifecycle_metadata[factor_name]
        current_status = lifecycle_meta.status
        
        # Check if transition is allowed
        if not self.transition_rules.is_transition_allowed(current_status, new_status):
            raise ValueError(f"Transition from {current_status.value} to {new_status.value} is not allowed")
        
        # Check prerequisites
        if not self._check_transition_conditions(factor_name, current_status, new_status):
            return False
        
        # Execute transition
        self._execute_transition(factor_name, new_status, reason, operator, metadata or {})
        
        # Save lifecycle metadata to database
        if self.factor_registry:
            self.factor_registry._handler.save_lifecycle_metadata(factor_name, lifecycle_meta)
        
        return True
    
    def _check_transition_conditions(self, factor_name: str, 
                                   from_status: FactorLifecycleStatus,
                                   to_status: FactorLifecycleStatus) -> bool:
        """Check transition prerequisites"""
        # All transitions are allowed by default (no special conditions)
        return True
    
    def _execute_transition(self, factor_name: str, new_status: FactorLifecycleStatus, reason: str, operator: str, metadata: Dict):
        """Execute status transition"""
        lifecycle_meta = self._lifecycle_metadata[factor_name]
        old_status = lifecycle_meta.status
        
        # Update status
        lifecycle_meta.status = new_status
        
        # Record transition history
        transition_record = StatusTransitionRecord(
            from_status=old_status,
            to_status=new_status,
            timestamp=datetime.datetime.now(),
            operator=operator,
            reason=reason,
            metadata=metadata
        )
        lifecycle_meta.status_history.append(transition_record)
        
        # Automatically trigger related operations
        self._handle_status_side_effects(factor_name, new_status)
        
        self._logger.info(f"Factor '{factor_name}' transitioned from {old_status.value} to {new_status.value}")
    
    def _handle_status_side_effects(self, factor_name: str, new_status: FactorLifecycleStatus):
        """Handle status transition side effects"""
        lifecycle_meta = self._lifecycle_metadata[factor_name]
        
        if new_status == FactorLifecycleStatus.ACTIVE:
            # Enable monitoring when factor becomes active
            lifecycle_meta.monitoring_enabled = True
            self._logger.info(f"Enabled monitoring for factor '{factor_name}'")
        
        elif new_status == FactorLifecycleStatus.DEPRECATED:
            # Send notification when deprecated
            self._send_deprecation_notice(factor_name)
    
    def create_version(self, factor_name: str, version_type: str = "patch", 
                      source_code: str = "", metadata: Dict = None,
                      operator: str = "system") -> FactorVersion:
        """Create new version"""
        if factor_name not in self._lifecycle_metadata:
            self.initialize_factor_lifecycle(factor_name)
        
        lifecycle_meta = self._lifecycle_metadata[factor_name]
        current_version = lifecycle_meta.version
        
        # Increment version number based on version type
        if version_type == "major":
            new_version = current_version.increment_major()
        elif version_type == "minor":
            new_version = current_version.increment_minor()
        else:  # patch
            new_version = current_version.increment_patch()
        
        # Create version record
        self.version_manager.create_version(factor_name, new_version, source_code, metadata)
        
        # Update lifecycle metadata
        lifecycle_meta.version = new_version
        lifecycle_meta.version_history.append({
            "version": str(new_version),
            "created_at": datetime.datetime.now().isoformat(),
            "operator": operator,
            "parent_version": str(current_version)
        })
        
        self._logger.info(f"Created version {new_version} for factor '{factor_name}'")
        return new_version
    
    def get_lifecycle_info(self, factor_name: str) -> Optional[LifecycleMetadata]:
        """Get lifecycle information"""
        return self._lifecycle_metadata.get(factor_name)
    
    def get_factors_by_status(self, status: FactorLifecycleStatus) -> List[str]:
        """Get factor list by status"""
        factors = []
        for name, metadata in self._lifecycle_metadata.items():
            if metadata.status == status:
                factors.append(name)
        return factors
    
    def _send_deprecation_notice(self, factor_name: str):
        """Send deprecation notice"""
        lifecycle_meta = self._lifecycle_metadata[factor_name]
        contacts = lifecycle_meta.alert_contacts
        
        # Add actual notification logic here
        self._logger.warning(f"Factor '{factor_name}' has been deprecated. Contacts: {contacts}")
    
    def export_lifecycle_data(self, factor_name: str = None) -> Dict:
        """Export lifecycle data"""
        if factor_name:
            if factor_name in self._lifecycle_metadata:
                return self._serialize_lifecycle_metadata(self._lifecycle_metadata[factor_name])
            return {}
        else:
            result = {}
            for name, metadata in self._lifecycle_metadata.items():
                result[name] = self._serialize_lifecycle_metadata(metadata)
            return result
    
    def _serialize_lifecycle_metadata(self, metadata: LifecycleMetadata) -> Dict:
        """Serialize lifecycle metadata"""
        result = {
            "status": metadata.status.value,
            "version": str(metadata.version),
            "status_history": [],
            "monitoring_enabled": metadata.monitoring_enabled,
            "alert_contacts": metadata.alert_contacts,
            "version_history": metadata.version_history,
            "parent_version": metadata.parent_version
        }
        
        # Serialize status history
        for record in metadata.status_history:
            result["status_history"].append({
                "from_status": record.from_status.value,
                "to_status": record.to_status.value,
                "timestamp": record.timestamp.isoformat(),
                "operator": record.operator,
                "reason": record.reason,
                "metadata": record.metadata
            })
        
        return result