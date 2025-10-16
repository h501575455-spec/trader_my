import uuid
import hashlib
import datetime
import re
from typing import Optional, Set
from dataclasses import dataclass


@dataclass
class FactorUID:
    """Factor unique identifier"""
    uid: str
    name: str
    created_at: datetime.datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
    
    def __str__(self):
        return self.uid
    
    def __hash__(self):
        return hash(self.uid)
    
    def __eq__(self, other):
        if isinstance(other, FactorUID):
            return self.uid == other.uid
        elif isinstance(other, str):
            return self.uid == other
        return False


class UIDGenerationStrategy:
    """Base class for UID generation strategies"""
    
    def generate(self, name: str, **kwargs) -> str:
        """Generate unique ID"""
        raise NotImplementedError("Subclasses must implement generate method")
    
    def validate_name(self, name: str) -> bool:
        """Validate factor name format"""
        # Basic rule: starts with letter, contains only letters, numbers, underscores and hyphens
        pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
        return bool(re.match(pattern, name)) and len(name) >= 3 and len(name) <= 100


class UUIDStrategy(UIDGenerationStrategy):
    """Generate unique ID using UUID4"""
    
    def generate(self, name: str, **kwargs) -> str:
        """Generate UUID4-based unique ID"""
        unique_id = str(uuid.uuid4())
        return unique_id.replace('-', '_')


class NamespaceUUIDStrategy(UIDGenerationStrategy):
    """Generate unique ID using UUID5"""

    def __init__(self, base_namespace: str = "frozen.factor"):
        self.base_namespace = base_namespace
        self.namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, base_namespace)

    def generate(self, name: str, **kwargs) -> str:
        """Generate namespace-based unique ID"""
        unique_id = uuid.uuid5(self.namespace_uuid, name)
        return str(unique_id).replace('-', '_')


class HashBasedStrategy(UIDGenerationStrategy):
    """Hash-based unique ID generation strategy"""
    
    def generate(self, name: str,
                 author: str = "", timestamp: datetime.datetime = None, **kwargs) -> str:
        """Generate hash-based unique ID from multiple fields"""
        if timestamp is None:
            timestamp = datetime.datetime.now()

        # Build hash source string
        hash_source = f"{name}:{author}:{timestamp.isoformat()}"
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(hash_source.encode())
        hash_hex = hash_object.hexdigest()
        
        # Use first 16 characters as unique ID
        return hash_hex[:16]


class HumanReadableStrategy(UIDGenerationStrategy):
    """Human-readable unique ID generation strategy"""
    
    def __init__(self, separator: str = "_"):
        self.separator = separator
    
    def generate(self, name: str,
                 author: str = "", **kwargs) -> str:
        """Generate human-readable unique ID"""
        # Clean name: convert to lowercase, replace special characters
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())

        # Generate timestamp (YYYYMMDD_HHMM)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        # Generate short UUID as suffix to ensure uniqueness
        short_uuid = str(uuid.uuid4())[:8]

        components = [clean_name, timestamp, short_uuid]
        return self.separator.join(components)


class SequentialStrategy(UIDGenerationStrategy):
    """Sequential number-based unique ID generation strategy"""
    
    def __init__(self, prefix: str = "", start_number: int = 1):
        self.prefix = prefix
        self.current_number = start_number
        self._used_numbers: Set[int] = set()
    
    def generate(self, name: str, **kwargs) -> str:
        """Generate sequential number-based unique ID"""
        # Find next available sequence number
        while self.current_number in self._used_numbers:
            self.current_number += 1

        # Record used sequence number
        self._used_numbers.add(self.current_number)
        
        # Generate ID
        if self.prefix:
            uid = f"{self.prefix}_{self.current_number:06d}"
        else:
            uid = f"{self.current_number:06d}"
        self.current_number += 1
        
        return uid
    
    def set_used_numbers(self, used_numbers: Set[int]):
        """Set used sequence numbers (used when loading from database)"""
        self._used_numbers.update(used_numbers)
        if used_numbers:
            self.current_number = max(used_numbers) + 1


class UIDManager:
    """UID manager responsible for generating and validating unique IDs"""
    
    def __init__(self, strategy: UIDGenerationStrategy = None):
        self.strategy = strategy or UUIDStrategy()
        self._registered_uids: Set[str] = set()
        self._name_to_uid: dict = {}  # name -> uid mapping
        self._uid_to_name: dict = {}  # uid -> name mapping
        self._uid_objects: dict = {}  # uid -> FactorUID object mapping
    
    def generate_uid(self, name: str, **kwargs) -> FactorUID:
        """Generate new factor unique ID"""
        # Validate name format
        if not self.strategy.validate_name(name):
            raise ValueError(f"Invalid factor name format: '{name}'. "
                           "Name must start with letter, contain only letters, numbers, "
                           "underscores and hyphens, and be 3-100 characters long.")
        
        # Check if name already exists
        if name in self._name_to_uid:
            raise ValueError(f"Factor name '{name}' already exists with UID: {self._name_to_uid[name]}")
        
        # Generate unique ID
        max_attempts = 100
        for attempt in range(max_attempts):
            uid = self.strategy.generate(name=name, **kwargs)

            if uid not in self._registered_uids:
                # Register new UID
                factor_uid = FactorUID(uid=uid, name=name)
                self._registered_uids.add(uid)
                self._name_to_uid[name] = uid
                self._uid_to_name[uid] = name
                self._uid_objects[uid] = factor_uid
                return factor_uid
        
        raise RuntimeError(f"Failed to generate unique UID for '{name}' after {max_attempts} attempts")
    
    def register_existing_uid(self, uid: str, name: str):
        """Register existing UID (used for loading from database)"""
        if uid in self._registered_uids:
            raise ValueError(f"UID '{uid}' already registered")

        if name in self._name_to_uid:
            raise ValueError(f"Factor name '{name}' already registered with UID: {self._name_to_uid[name]}")

        factor_uid = FactorUID(uid=uid, name=name)
        self._registered_uids.add(uid)
        self._name_to_uid[name] = uid
        self._uid_to_name[uid] = name
        self._uid_objects[uid] = factor_uid
    
    def is_name_available(self, name: str) -> bool:
        """Check if name is available"""
        return name not in self._name_to_uid
    
    def is_uid_available(self, uid: str) -> bool:
        """Check if UID is available"""
        return uid not in self._registered_uids
    
    def get_uid_by_name(self, name: str) -> Optional[str]:
        """Get UID by name"""
        return self._name_to_uid.get(name)
    
    def get_name_by_uid(self, uid: str) -> Optional[str]:
        """Get name by UID"""
        return self._uid_to_name.get(uid)
    
    def remove_factor(self, name: str = None, uid: str = None):
        """Remove factor's UID registration"""
        if name:
            uid = self._name_to_uid.get(name)
            if uid:
                self._registered_uids.discard(uid)
                del self._name_to_uid[name]
                del self._uid_to_name[uid]
                if uid in self._uid_objects:
                    del self._uid_objects[uid]
        elif uid:
            name = self._uid_to_name.get(uid)
            if name:
                self._registered_uids.discard(uid)
                del self._name_to_uid[name]
                del self._uid_to_name[uid]
                if uid in self._uid_objects:
                    del self._uid_objects[uid]
    
    def get_all_uids(self) -> Set[str]:
        """Get all registered UIDs"""
        return self._registered_uids.copy()
    
    def get_all_names(self) -> Set[str]:
        """Get all registered names"""
        return set(self._name_to_uid.keys())
    
    def get_statistics(self) -> dict:
        """Get UID manager statistics"""
        return {
            "total_registered": len(self._registered_uids),
            "strategy": self.strategy.__class__.__name__
        }

    def bulk_register_existing(self, uid_name_pairs: list):
        """Bulk register existing UIDs (used for database migration)"""
        for uid, name in uid_name_pairs:
            try:
                self.register_existing_uid(uid, name)
            except ValueError as e:
                print(f"Warning: Failed to register {uid}:{name} - {e}")
    
    def validate_uid_format(self, uid: str) -> bool:
        """Validate UID format"""
        # Check for different UID formats produced by our strategies
        
        # UUID format (32 hex chars with underscores, length 36): xxxxxxxx_xxxx_xxxx_xxxx_xxxxxxxxxxxx
        uuid_pattern = r'^[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$'
        if re.match(uuid_pattern, uid):
            return True
        
        # Hash format (16 hex chars, length 16): xxxxxxxxxxxxxxxx
        hash_pattern = r'^[0-9a-f]{16}$'
        if re.match(hash_pattern, uid):
            return True
        
        # Sequential format (6-digit numbers): 000001 or prefix_000001
        sequential_pattern = r'^(\w+_)?\d{6}$'
        if re.match(sequential_pattern, uid):
            return True
        
        # Human readable format: name_YYYYMMDD_HHMM_xxxxxxxx
        # This is complex: contains at least 3 parts separated by underscores,
        # has a timestamp pattern YYYYMMDD_HHMM, and ends with 8-char hex
        human_readable_pattern = r'^[a-z0-9_-]+_\d{8}_\d{4}_[0-9a-f]{8}$'
        if re.match(human_readable_pattern, uid):
            return True
        
        # If none of the specific UID patterns match, it's not a valid UID
        return False


# Predefined strategy instances
DEFAULT_STRATEGIES = {
    'uuid': UUIDStrategy(),
    'namespace_uuid': NamespaceUUIDStrategy(),
    'hash': HashBasedStrategy(),
    'human_readable': HumanReadableStrategy(),
    'sequential': SequentialStrategy()
}


def create_uid_manager(strategy_name: str = 'uuid') -> UIDManager:
    """Create UID manager"""
    if strategy_name not in DEFAULT_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available strategies: {list(DEFAULT_STRATEGIES.keys())}")
    
    strategy = DEFAULT_STRATEGIES[strategy_name]
    return UIDManager(strategy)