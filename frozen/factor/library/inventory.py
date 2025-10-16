import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import Dict, List, Optional

from .lifecycle import (
    FactorLifecycleStatus, FactorVersion, LifecycleMetadata,
    FactorLifecycleManager, FactorVersionManager
)

from .uid_manager import FactorUID, UIDManager, create_uid_manager

# Try to import networkx, but make it optional
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False

from ..expression.base import Factor
from ...data.database import DatabaseTypes
from .handlers.base import FactorHandler
from .handlers.mongo import MongoFactorHandler
from .handlers.duck import DuckDBFactorHandler
from ...utils.log import GL


@dataclass
class FactorMetadata:
    """Factor metadata class"""
    # Basic fields
    name: str
    uid: str = ""  # Unique identifier for the factor
    description: str = ""
    dependencies: List[str] = None  # List of dependency UIDs
    category: str = "default"
    tags: List[str] = None
    created_time: datetime.datetime = None
    updated_time: datetime.datetime = None
    
    # Lifecycle management fields
    lifecycle_status: FactorLifecycleStatus = FactorLifecycleStatus.DEVELOPMENT
    version: FactorVersion = None
    author: str = ""
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.created_time is None:
            self.created_time = datetime.datetime.now()
        if self.updated_time is None:
            self.updated_time = datetime.datetime.now()
        if self.version is None:
            self.version = FactorVersion()
    
    def update_timestamp(self):
        """Update the modification timestamp"""
        self.updated_time = datetime.datetime.now()
    
    def is_production_ready(self) -> bool:
        """Check if factor is ready for production use"""
        return self.lifecycle_status == FactorLifecycleStatus.ACTIVE
    
    def needs_attention(self) -> bool:
        """Check if factor needs attention"""
        # In simplified model, no specific status indicates need for attention
        # This could be extended based on performance metrics or other criteria
        return False
    
    def get_full_identifier(self) -> str:
        """Get full identifier"""
        return f"{self.name}#{self.uid}"
    
    def validate_dependencies(self, available_uids: set) -> List[str]:
        """Validate that all dependencies exist and return missing ones"""
        missing = []
        for dep_uid in self.dependencies:
            if dep_uid not in available_uids:
                missing.append(dep_uid)
        return missing


class FactorRegistry:
    """Factor registry for managing factor definitions and dependencies"""
    
    def __init__(self, handler=None, enable_lifecycle=True, uid_strategy="uuid"):
        self._factors: Dict[str, FactorMetadata] = {}  # uid -> metadata mapping
        self._handler = handler
        if HAS_NETWORKX:
            self._dependency_graph = nx.DiGraph()
        else:
            self._dependency_graph = None
        self._logger = GL.get_logger(__name__)
        
        # Initialize UID management
        self.uid_manager = create_uid_manager(uid_strategy)
        
        # Initialize lifecycle management
        self.enable_lifecycle = enable_lifecycle
        if enable_lifecycle:
            self.lifecycle_manager = FactorLifecycleManager(
                factor_registry=self,
                version_manager=FactorVersionManager()
            )
        else:
            self.lifecycle_manager = None
        
        # Load existing metadata from database if handler is provided
        if self._handler:
            self._load_from_database()
    
    def register_factor(self, name: str, description: str = "",
                       dependencies: List[str] = None,
                       category: str = "default",
                       tags: List[str] = None,
                       author: str = "",
                       version: FactorVersion = None,
                       initial_status: FactorLifecycleStatus = FactorLifecycleStatus.DEVELOPMENT) -> str:
        """Register a factor with its metadata
        
        Returns:
            str: The generated unique ID for the factor
            
        Raises:
            ValueError: If factor name already exists or is invalid
        """
        if dependencies is None:
            dependencies = []
        
        # Generate unique ID and validate name
        try:
            factor_uid = self.uid_manager.generate_uid(
                name=name,
                author=author,
                timestamp=datetime.datetime.now()
            )
        except ValueError as e:
            raise ValueError(f"Failed to register factor '{name}': {e}")
        
        # Validate dependencies exist (convert name-based to UID-based if needed)
        validated_dependencies = []
        for dep in dependencies:
            # Check if dependency is a name or UID
            if self._is_uid(dep):
                # It's already a UID
                if dep not in self._factors:
                    raise ValueError(f"Dependency UID '{dep}' not found for factor '{name}'")
                validated_dependencies.append(dep)
            else:
                # It's a name, convert to UID
                dep_uid = self.uid_manager.get_uid_by_name(dep)
                if not dep_uid:
                    raise ValueError(f"Dependency '{dep}' not found for factor '{name}'")
                validated_dependencies.append(dep_uid)
        
        # Create enhanced metadata
        metadata = FactorMetadata(
            name=name,
            uid=factor_uid.uid,
            description=description,
            dependencies=validated_dependencies,
            category=category,
            tags=tags or [],
            author=author,
            version=version or FactorVersion(),
            lifecycle_status=initial_status
        )
        
        # Store by UID (not name)
        self._factors[factor_uid.uid] = metadata
        
        # Initialize lifecycle if enabled
        if self.lifecycle_manager:
            self.lifecycle_manager.initialize_factor_lifecycle(
                factor_uid.uid, initial_status, author or "system"
            )
        
        # Save to database if handler is provided
        if self._handler:
            self._handler.save_factor_metadata(metadata)
        
        # Update dependency graph
        if self._dependency_graph is not None:
            self._dependency_graph.add_node(factor_uid.uid)
            for dep_uid in validated_dependencies:
                self._dependency_graph.add_edge(dep_uid, factor_uid.uid)
            
            # Check for circular dependencies
            if not nx.is_directed_acyclic_graph(self._dependency_graph):
                # Remove the problematic factor
                self._dependency_graph.remove_node(factor_uid.uid)
                del self._factors[factor_uid.uid]
                self.uid_manager.remove_factor(uid=factor_uid.uid)
                # Also remove from database
                if self._handler:
                    self._handler.delete_factor_metadata(name)
                raise ValueError(f"Circular dependency detected when adding factor '{name}'")
        
        self._logger.info(f"Factor '{name}' registered successfully with UID {factor_uid.uid} and status {initial_status.value}")
        return factor_uid.uid
    
    def _is_uid(self, identifier: str) -> bool:
        """Check if string is a UID format"""
        return self.uid_manager.validate_uid_format(identifier)
    
    def get_factor_by_name(self, name: str) -> Optional[FactorMetadata]:
        """Get factor metadata by name"""
        uid = self.uid_manager.get_uid_by_name(name)
        if uid and uid in self._factors:
            return self._factors[uid]
        return None
    
    def get_factor_by_uid(self, uid: str) -> Optional[FactorMetadata]:
        """Get factor metadata by UID"""
        return self._factors.get(uid)
    
    def get_factor_info(self, identifier: str) -> FactorMetadata:
        """Get factor metadata by name or UID"""
        # Try as UID first
        if self._is_uid(identifier):
            factor = self.get_factor_by_uid(identifier)
            if factor:
                return factor
        
        # Try as name
        factor = self.get_factor_by_name(identifier)
        if factor:
            return factor
        
        raise ValueError(f"Factor '{identifier}' not registered")
    
    def check_name_available(self, name: str) -> bool:
        """Check if factor name is available"""
        return self.uid_manager.is_name_available(name)
    
    def get_execution_order(self, factor_identifiers: List[str] = None) -> List[str]:
        """Get factors in topological order for execution
        
        Args:
            factor_identifiers: List of factor names or UIDs. If None, returns all factors.
            
        Returns:
            List of factor UIDs in execution order
        """
        if factor_identifiers is None:
            factor_uids = list(self._factors.keys())
        else:
            # Convert names/UIDs to UIDs
            factor_uids = []
            for identifier in factor_identifiers:
                if self._is_uid(identifier):
                    if identifier not in self._factors:
                        raise ValueError(f"Factor UID '{identifier}' not registered")
                    factor_uids.append(identifier)
                else:
                    uid = self.uid_manager.get_uid_by_name(identifier)
                    if not uid:
                        raise ValueError(f"Factor name '{identifier}' not registered")
                    factor_uids.append(uid)
        
        if self._dependency_graph is None:
            # Simple dependency resolution without networkx
            return self._simple_topological_sort(factor_uids)
        
        # Create subgraph with only requested factors and their dependencies
        subgraph_nodes = set()
        for uid in factor_uids:
            subgraph_nodes.add(uid)
            # Add all dependencies recursively
            subgraph_nodes.update(nx.ancestors(self._dependency_graph, uid))
        
        subgraph = self._dependency_graph.subgraph(subgraph_nodes)
        return list(nx.topological_sort(subgraph))
    
    def list_factors(self, category: str = None, tags: List[str] = None,
                     status: FactorLifecycleStatus = None, production_ready_only: bool = False, return_uids: bool = True) -> List[str]:
        """List registered factors with filtering options
        
        Args:
            category: Filter by category
            tags: Filter by tags (must have any of these tags)
            status: Filter by lifecycle status
            production_ready_only: Only return production-ready factors
            return_uids: If True, return UIDs; if False, return names
            
        Returns:
            List of factor UIDs or names
        """
        factors = []
        for uid, metadata in self._factors.items():
            if category and metadata.category != category:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            if status and getattr(metadata, "lifecycle_status", FactorLifecycleStatus.DEVELOPMENT) != status:
                continue
            if production_ready_only and not getattr(metadata, "is_production_ready", lambda: False)():
                continue
            
            if return_uids:
                factors.append(uid)
            else:
                factors.append(metadata.name)
        
        return factors
    
    def remove_factor(self, identifier: str, force: bool = False, operator: str = "system") -> None:
        """Remove a factor from registry
        
        Args:
            identifier: Factor name or UID
            force: Force removal even if factor is active or has dependents
            operator: User performing the operation
        """
        # Get factor metadata
        factor = self.get_factor_info(identifier)
        if not factor:
            raise ValueError(f"Factor '{identifier}' not registered")
        
        factor_uid = factor.uid
        factor_name = factor.name
        
        # Check lifecycle status before removal
        if factor.lifecycle_status == FactorLifecycleStatus.ACTIVE and not force:
            raise ValueError(f"Cannot remove active factor '{factor_name}' without force=True")
        
        # Archive the factor before removal if lifecycle is enabled
        if self.lifecycle_manager and not force:
            try:
                self.lifecycle_manager.transition_status(
                    factor_uid, FactorLifecycleStatus.ARCHIVED, 
                    "Factor Removal", operator
                )
            except Exception as e:
                self._logger.warning(f"Failed to archive factor before removal: {e}")
        
        # Check if other factors depend on this one
        if self._dependency_graph is not None:
            dependents = list(self._dependency_graph.successors(factor_uid))
            if dependents and not force:
                dependent_names = [self._factors[dep_uid].name for dep_uid in dependents if dep_uid in self._factors]
                raise ValueError(f"Cannot remove factor '{factor_name}': factors {dependent_names} depend on it")
            self._dependency_graph.remove_node(factor_uid)
        else:
            # Manual check for dependencies
            dependents = []
            for uid, metadata in self._factors.items():
                if factor_uid in metadata.dependencies:
                    dependents.append(metadata.name)
            if dependents and not force:
                raise ValueError(f"Cannot remove factor '{factor_name}': factors {dependents} depend on it")
        
        # Remove from registry and UID manager
        del self._factors[factor_uid]
        self.uid_manager.remove_factor(name=factor_name, uid=factor_uid)
        
        # Remove from database if handler is provided
        if self._handler:
            self._handler.delete_factor_metadata(factor_name)
        
        self._logger.info(f"Factor '{factor_name}' (UID: {factor_uid}) removed successfully")
    
    def _simple_topological_sort(self, factor_uids: List[str]) -> List[str]:
        """Simple topological sort without networkx using UIDs"""
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(uid: str):
            if uid in temp_visited:
                factor_name = self._factors[uid].name if uid in self._factors else uid
                raise ValueError(f"Circular dependency detected involving factor '{factor_name}'")
            if uid in visited:
                return
            
            temp_visited.add(uid)
            if uid in self._factors:
                for dep_uid in self._factors[uid].dependencies:
                    visit(dep_uid)
            temp_visited.remove(uid)
            visited.add(uid)
            result.append(uid)
        
        for uid in factor_uids:
            if uid not in visited:
                visit(uid)
        
        return result
    
    def _load_from_database(self):
        """Load factor metadata from database and rebuild UID mappings"""
        try:
            loaded_factors = self._handler.load_all_factor_metadata()
            
            # Rebuild UID mappings
            uid_name_pairs = []
            for identifier, metadata in loaded_factors.items():
                # Handle legacy data where factors might be stored by name
                if hasattr(metadata, "uid") and metadata.uid:
                    # Modern format with UID - collect for bulk registration
                    uid_name_pairs.append((metadata.uid, metadata.name))
                    # Store by UID
                    self._factors[metadata.uid] = metadata
                else:
                    # Legacy format, generate UID for existing factor
                    try:
                        factor_uid = self.uid_manager.generate_uid(
                            name=metadata.name,
                            author=getattr(metadata, "author", ""),
                        )
                        metadata.uid = factor_uid.uid
                        # Note: generate_uid already registers the UID, so no need to add to uid_name_pairs
                        # Store by new UID
                        self._factors[factor_uid.uid] = metadata
                        # Update database with UID
                        if self._handler:
                            self._handler.save_factor_metadata(metadata)
                    except ValueError as e:
                        self._logger.error(f"Failed to generate UID for legacy factor '{metadata.name}': {e}")
                        continue
            
            # Register existing UIDs in bulk (only for factors loaded with UIDs from database)
            self.uid_manager.bulk_register_existing(uid_name_pairs)
            
            self._rebuild_dependency_graph()
            self._logger.info(f"Loaded {len(self._factors)} factors from database")
        except Exception as e:
            self._logger.warning(f"Failed to load factors from database: {e}")
            self._factors = {}
    
    def _rebuild_dependency_graph(self):
        """Rebuild dependency graph from loaded metadata"""
        if self._dependency_graph is None:
            return
        
        self._dependency_graph.clear()
        
        # Add all nodes first
        for uid in self._factors.keys():
            self._dependency_graph.add_node(uid)
        
        # Add edges based on dependencies
        for uid, metadata in self._factors.items():
            for dep_uid in metadata.dependencies:
                if dep_uid in self._factors:
                    self._dependency_graph.add_edge(dep_uid, uid)
                else:
                    factor_name = metadata.name
                    self._logger.warning(f"Factor '{factor_name}' depends on missing factor UID '{dep_uid}'")
    
    # Additional UID-related utility methods
    def get_factor_uid_by_name(self, name: str) -> Optional[str]:
        """Get factor UID by name"""
        return self.uid_manager.get_uid_by_name(name)
    
    def get_factor_name_by_uid(self, uid: str) -> Optional[str]:
        """Get factor name by UID"""
        return self.uid_manager.get_name_by_uid(uid)
    
    def get_all_factor_names(self) -> List[str]:
        """Get all registered factor names"""
        return [metadata.name for metadata in self._factors.values()]
    
    def get_all_factor_uids(self) -> List[str]:
        """Get all registered factor UIDs"""
        return list(self._factors.keys())
    
    def get_factor_summary(self) -> Dict:
        """Get summary of all factors with names and UIDs"""
        summary = {}
        for uid, metadata in self._factors.items():
            summary[metadata.name] = {
                "uid": uid,
                "status": metadata.lifecycle_status.value,
                "version": str(metadata.version),
                "category": metadata.category,
                "author": metadata.author
            }
        return summary
    
    def validate_all_dependencies(self) -> Dict[str, List[str]]:
        """Validate all factor dependencies and return missing ones"""
        validation_results = {}
        
        # Get all UIDs from both memory and database
        all_uids = set(self._factors.keys())  # Currently loaded factors
        
        # If we have a database handler, also check database for all existing factor UIDs
        if self._handler:
            try:
                all_metadata_from_db = self._handler.load_all_factor_metadata()
                for identifier, metadata in all_metadata_from_db.items():
                    if hasattr(metadata, "uid") and metadata.uid:
                        all_uids.add(metadata.uid)
                    # Also add the identifier itself in case it's a UID
                    if self._is_uid(identifier):
                        all_uids.add(identifier)
            except Exception as e:
                self._logger.warning(f"Failed to load factor metadata from database for dependency validation: {e}")
        
        # Validate dependencies for each factor
        for uid, metadata in self._factors.items():
            missing_deps = metadata.validate_dependencies(all_uids)
            if missing_deps:
                validation_results[metadata.name] = missing_deps
        
        return validation_results

    def visualize_dependency_graph(self, filename: str = "dependency_graph.png"):
        """Visualize the dependency graph using matplotlib and networkx"""
        if self._dependency_graph is None:
            self._logger.warning("networkx not available for visualization.")
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self._dependency_graph, k=2, iterations=50) # Increase spacing between nodes
        
        # Create a mapping from UID to name for labels
        uid_to_name = {}
        for uid in self._dependency_graph.nodes():
            factor_name = self.get_factor_name_by_uid(uid)
            uid_to_name[uid] = factor_name if factor_name else uid
        
        # Draw edges first (so they appear behind nodes)
        nx.draw_networkx_edges(self._dependency_graph, pos, 
                             edge_color="black", 
                             arrows=True, 
                             arrowsize=15,           # Larger arrows
                             arrowstyle="-|>",
                             width=1.5,             # Thicker edges
                             node_size=2500,        # Account for node size in edge drawing
                             alpha=0.8)             # Semi-transparent for better visibility
        
        # Draw nodes on top of edges
        nx.draw_networkx_nodes(self._dependency_graph, pos, 
                              node_size=2500,       # Slightly smaller nodes
                              node_color="lightblue",
                              edgecolors="gray",    # Node border for better definition
                              linewidths=2)
        
        # Draw labels using factor names
        nx.draw_networkx_labels(self._dependency_graph, pos, 
                               labels=uid_to_name,
                               font_size=9, 
                               font_weight="bold",
                               font_color="black")

        plt.title("Factor Dependency Graph", fontsize=14, fontweight="bold")
        plt.axis("off") # Hide axes
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Dependency graph visualized and saved to {filename}")

    def visualize_dependency_graph_advanced(
        self,
        filename: str = "dependency_graph_advanced.png",
        layout: str = "spring",
        show_categories: bool = True,
        node_size: int = 3000,
        figsize: tuple = (15, 10),
        interactive: bool = False,
        auto_open: bool = False
    ):
        """Advanced visualization with multiple layout options and category coloring
        
        Args:
            filename: Output filename
            layout: Layout algorithm ('spring', 'circular', 'shell', 'kamada_kawai', 'hierarchical')
            show_categories: Whether to color nodes by category
            node_size: Size of nodes
            figsize: Figure size for static plots
            interactive: If True, create interactive plotly visualization instead of static matplotlib
            auto_open: Whether to auto-open interactive plot
        """
        if self._dependency_graph is None:
            self._logger.warning("networkx not available for visualization.")
            return

        if interactive:
            return self._create_interactive_visualization(filename, layout, show_categories, node_size, auto_open)
        else:
            return self._create_static_visualization(filename, layout, show_categories, node_size, figsize)
    
    def _create_interactive_visualization(self, filename, layout, show_categories, node_size, auto_open):
        """Create interactive plotly visualization"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
        except ImportError:
            self._logger.error("Plotly not available. Please install with: pip install plotly")
            return
        
        # Choose layout algorithm
        if layout == "spring":
            pos = nx.spring_layout(self._dependency_graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self._dependency_graph)
        elif layout == "shell":
            pos = nx.shell_layout(self._dependency_graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self._dependency_graph)
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(self._dependency_graph, prog='dot')
            except:
                pos = nx.spring_layout(self._dependency_graph)
        else:
            pos = nx.spring_layout(self._dependency_graph)
        
        # Create UID to name mapping
        uid_to_name = {}
        for uid in self._dependency_graph.nodes():
            factor_name = self.get_factor_name_by_uid(uid)
            uid_to_name[uid] = factor_name if factor_name else uid
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        for edge in self._dependency_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#666'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        )
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_hover_text = []
        node_sizes = []
        
        # Category colors mapping
        category_colors = {
            "market_data": "#87CEEB",      # SkyBlue
            "derived": "#98FB98",          # PaleGreen
            "risk": "#F08080",             # LightCoral
            "technical": "#FFE4B5",        # Moccasin
            "composite": "#DDA0DD",        # Plum
            "basic": "#B0E0E6",           # PowderBlue
            "volume-price": "#F0E68C",     # Khaki
            "default": "#D3D3D3"           # LightGray
        }
        
        for node in self._dependency_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            factor_name = uid_to_name.get(node, node)
            node_text.append(factor_name)
            
            # Calculate node size based on text length
            text_length = len(factor_name)
            base_size = max(30, text_length * 2.5)  # Dynamic size based on text length
            node_sizes.append(min(base_size, 80))   # Cap maximum size
            
            # Get node metadata for hover and styling
            if node in self._factors:
                metadata = self._factors[node]
                
                # Create hover text with factor details
                hover_info = [
                    f"<b>{factor_name}</b>",
                    f"Category: {metadata.category}",
                    f"Description: {metadata.description[:100]}..." if len(metadata.description) > 100 else f"Description: {metadata.description}",
                    f"Author: {getattr(metadata, 'author', 'Unknown')}",
                    f"Status: {getattr(metadata, 'lifecycle_status', 'Unknown').value if hasattr(getattr(metadata, 'lifecycle_status', None), 'value') else getattr(metadata, 'lifecycle_status', 'Unknown')}",
                    f"Dependencies: {len(metadata.dependencies)}",
                    f"Created: {metadata.created_time.strftime('%Y-%m-%d') if metadata.created_time else 'Unknown'}"
                ]
                node_hover_text.append("<br>".join(hover_info))
                
                # Color by category if requested
                if show_categories:
                    node_colors.append(category_colors.get(metadata.category, category_colors["default"]))
                else:
                    node_colors.append(category_colors["default"])
            else:
                node_hover_text.append(f"<b>{factor_name}</b><br>No metadata available")
                node_colors.append(category_colors["default"])
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_hover_text,
            text=node_text,
            textposition="middle center",
            textfont=dict(
                size=10, 
                color="black", 
                family="Arial",
            ),
            marker=dict(
                size=node_sizes,  # Dynamic sizes
                color=node_colors,
                line=dict(width=2, color='#000'),
                opacity=0.8
            ),
            name='Factors'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Interactive Factor Dependency Graph ({layout} layout)',
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )
        
        # Add category legend if showing categories
        if show_categories:
            categories_used = set()
            for node in self._dependency_graph.nodes():
                if node in self._factors:
                    categories_used.add(self._factors[node].category)
            
            legend_text = "<br>".join([f"<span style='color: {category_colors.get(cat, category_colors['default'])};'>●</span> {cat}" 
                                     for cat in sorted(categories_used)])
            fig.add_annotation(
                text=f"<b>Categories:</b><br>{legend_text}",
                showarrow=False,
                x=1.02,
                y=1,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        # Convert .png to .html if needed
        if filename.endswith('.png'):
            filename = filename.replace('.png', '.html')
        
        # Save interactive plot
        pyo.plot(fig, filename=filename, auto_open=auto_open)
        self._logger.info(f"Interactive dependency graph saved to {filename}")
        return fig
    
    def _create_static_visualization(self, filename, layout, show_categories, node_size, figsize):
        """Create static matplotlib visualization"""
        plt.figure(figsize=figsize)
        
        # Choose layout algorithm
        if layout == "spring":
            pos = nx.spring_layout(self._dependency_graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self._dependency_graph)
        elif layout == "shell":
            pos = nx.shell_layout(self._dependency_graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self._dependency_graph)
        elif layout == "hierarchical":
            pos = nx.hierarchical_layout(self._dependency_graph)
        else:
            pos = nx.spring_layout(self._dependency_graph)

        # Prepare node colors by category if requested
        if show_categories:
            categories = {}
            colors = []
            for node in self._dependency_graph.nodes():
                if node in self._factors:
                    category = self._factors[node].category
                    if category not in categories:
                        categories[category] = len(categories)
                    colors.append(categories[category])
            
            # Create color map
            cmap = plt.cm.Set3
            node_colors = [cmap(colors[i] % cmap.N) for i in range(len(colors))]

        # Draw edges first (so they appear behind nodes)
        nx.draw_networkx_edges(self._dependency_graph, pos, 
                             edge_color="black", 
                             arrows=True, 
                             arrowsize=15,           # Larger arrows
                             arrowstyle="-|>",
                             width=1.5,             # Thicker edges
                             node_size=node_size,   # Account for node size
                             alpha=0.8)             # Semi-transparent

        # Draw nodes on top of edges
        if show_categories:
            # Draw nodes with category colors
            nx.draw_networkx_nodes(self._dependency_graph, pos, 
                                 node_color=node_colors, 
                                 node_size=node_size,
                                 edgecolors="black",    # Node border for definition
                                 linewidths=1.5)
            
            # Create legend for categories
            legend_elements = []
            for category, color_idx in categories.items():
                legend_elements.append(mpatches.Patch(color=cmap(color_idx % cmap.N), 
                                                   label=category))
            plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
        else:
            nx.draw_networkx_nodes(self._dependency_graph, pos, 
                                 node_color="lightblue", 
                                 node_size=node_size,
                                 edgecolors="gray",     # Node border for definition
                                 linewidths=1.5)

        # Create a mapping from UID to name for labels
        uid_to_name = {}
        for uid in self._dependency_graph.nodes():
            factor_name = self.get_factor_name_by_uid(uid)
            uid_to_name[uid] = factor_name if factor_name else uid

        # Draw labels using factor names
        nx.draw_networkx_labels(self._dependency_graph, pos, 
                              labels=uid_to_name,
                              font_size=9, 
                              font_weight="bold")

        plt.title(f"Factor Dependency Graph ({layout} layout)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Advanced dependency graph saved to {filename}")

    def visualize_subgraph(self, factor_names: List[str], 
                          filename: str = "subgraph.png",
                          include_dependencies: bool = True):
        """Visualize a subgraph containing specific factors and their dependencies"""
        if self._dependency_graph is None:
            self._logger.warning("networkx not available for visualization.")
            return

        # Convert factor names to UIDs if needed (for dependency graph lookup)
        factor_uids = []
        for name_or_uid in factor_names:
            if self._is_uid(name_or_uid):
                factor_uids.append(name_or_uid)
            else:
                # It's a name, get the UID
                uid = self.get_factor_uid_by_name(name_or_uid)
                if uid:
                    factor_uids.append(uid)
                else:
                    self._logger.warning(f"Factor '{name_or_uid}' not found")

        # Create subgraph using UIDs
        subgraph_nodes = set(factor_uids)
        if include_dependencies:
            for uid in factor_uids:
                if uid in self._dependency_graph:
                    subgraph_nodes.update(nx.ancestors(self._dependency_graph, uid))
        
        subgraph = self._dependency_graph.subgraph(subgraph_nodes)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=2, iterations=50)  # Better spacing
        
        # Color nodes: target factors in red, dependencies in blue
        node_colors = []
        for node in subgraph.nodes():
            if node in factor_uids:
                node_colors.append("pink")
            else:
                node_colors.append("lightblue")
        
        # Create a mapping from UID to name for labels
        uid_to_name = {}
        for uid in subgraph.nodes():
            factor_name = self.get_factor_name_by_uid(uid)
            uid_to_name[uid] = factor_name if factor_name else uid
        
        # Draw edges first (so they appear behind nodes)
        nx.draw_networkx_edges(subgraph, pos,
                              edge_color="black",
                              arrows=True,
                              arrowsize=15,          # Larger arrows
                              arrowstyle="-|>",
                              width=1.5,            # Thicker edges
                              node_size=2800,       # Account for node size
                              alpha=0.8)            # Semi-transparent
        
        # Draw nodes on top of edges
        nx.draw_networkx_nodes(subgraph, pos, 
                              node_color=node_colors,
                              node_size=2800,
                              edgecolors="gray",  # Node border for target factors
                              linewidths=2)
        
        # Draw labels using factor names
        nx.draw_networkx_labels(subgraph, pos,
                               labels=uid_to_name,
                               font_size=9,
                               font_weight="bold",
                               font_color="white")   # White text for better contrast
        
        # Convert UIDs back to names for title
        display_names = []
        for name_or_uid in factor_names:
            if self._is_uid(name_or_uid):
                name = self.get_factor_name_by_uid(name_or_uid)
                display_names.append(name if name else name_or_uid)
            else:
                display_names.append(name_or_uid)
        
        plt.title(f"Subgraph for factors: {', '.join(display_names)}")
        plt.axis("off")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        self._logger.info(f"Subgraph visualization saved to {filename}")

    def export_graph_data(self, filename: str = "dependency_graph.json"):
        """Export dependency graph data to JSON format for external visualization tools"""
        if self._dependency_graph is None:
            self._logger.warning("No dependency graph available.")
            return
        
        import json
        
        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {}
        }
        
        # Add nodes with metadata
        for node in self._dependency_graph.nodes():
            node_data = {"id": node}
            if node in self._factors:
                metadata = self._factors[node]
                node_data.update({
                    "description": metadata.description,
                    "category": metadata.category,
                    "tags": metadata.tags,
                    "dependencies": metadata.dependencies
                })
            graph_data["nodes"].append(node_data)
        
        # Add edges
        for edge in self._dependency_graph.edges():
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1]
            })
        
        # Add overall metadata
        graph_data["metadata"] = {
            "total_factors": len(self._factors),
            "total_dependencies": len(self._dependency_graph.edges()),
            "categories": list(set(m.category for m in self._factors.values())),
            "export_time": datetime.datetime.now().isoformat()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        self._logger.info(f"Graph data exported to {filename}")

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the dependency graph"""
        if self._dependency_graph is None:
            return {"error": "No dependency graph available"}
        
        stats = {
            "total_factors": len(self._factors),
            "total_nodes": self._dependency_graph.number_of_nodes(),
            "total_edges": self._dependency_graph.number_of_edges(),
            "categories": {},
            "dependency_stats": {}
        }
        
        # Category statistics
        for metadata in self._factors.values():
            category = metadata.category
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1
        
        # Dependency statistics
        in_degrees = dict(self._dependency_graph.in_degree())
        out_degrees = dict(self._dependency_graph.out_degree())
        
        stats["dependency_stats"] = {
            "max_in_degree": max(in_degrees.values()) if in_degrees else 0,
            "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
            "avg_in_degree": sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0,
            "isolated_nodes": len([n for n in self._dependency_graph.nodes() 
                                 if self._dependency_graph.degree(n) == 0])
        }
        
        return stats
    
    # Lifecycle management methods
    def transition_factor_status(self, name: str, new_status: FactorLifecycleStatus,
                               reason: str = "", operator: str = "system") -> bool:
        """Transition factor lifecycle status"""
        if not self.lifecycle_manager:
            raise RuntimeError("Lifecycle management is not enabled")
        
        if not self._handler:
            raise RuntimeError("No database handler available")
        
        # Load factor metadata from database
        factor_metadata = self._handler.load_factor_metadata(name)
        if not factor_metadata:
            raise ValueError(f"Factor '{name}' not found in database")
        
        success = self.lifecycle_manager.transition_status(name, new_status, reason, operator)
        if success:
            # Update metadata from database
            factor_metadata.lifecycle_status = new_status
            factor_metadata.update_timestamp()
            
            # Save updated metadata back to database
            try:
                self._handler.save_factor_metadata(factor_metadata)
                # Update memory cache if factor is loaded
                factor_uid = self.uid_manager.get_uid_by_name(name)
                if factor_uid and factor_uid in self._factors:
                    self._factors[factor_uid] = factor_metadata
            except Exception as e:
                self._logger.error(f"Failed to save updated metadata for factor '{name}': {e}")
        
        return success
    
    def create_factor_version(self, name: str, version_type: str = "patch",
                            source_code: str = "", metadata: Dict = None,
                            operator: str = "system") -> FactorVersion:
        """Create a new version of a factor"""
        if not self.lifecycle_manager:
            raise RuntimeError("Lifecycle management is not enabled")
        
        if not self._handler:
            raise RuntimeError("No database handler available")
        
        # Load factor metadata from database
        factor_metadata = self._handler.load_factor_metadata(name)
        if not factor_metadata:
            raise ValueError(f"Factor '{name}' not found in database")
        
        new_version = self.lifecycle_manager.create_version(
            name, version_type, source_code, metadata, operator
        )
        
        # Update metadata from database
        factor_metadata.version = new_version
        factor_metadata.update_timestamp()
        
        # Save updated metadata back to database
        try:
            self._handler.save_factor_metadata(factor_metadata)
            # Update memory cache if factor is loaded
            factor_uid = self.uid_manager.get_uid_by_name(name)
            if factor_uid and factor_uid in self._factors:
                self._factors[factor_uid] = factor_metadata
        except Exception as e:
            self._logger.error(f"Failed to save updated metadata for factor '{name}': {e}")
        
        return new_version
    
    def get_lifecycle_info(self, name: str) -> Optional[Dict]:
        """Get comprehensive lifecycle information for a factor"""
        if not self.lifecycle_manager:
            return None
        
        lifecycle_meta = self.lifecycle_manager.get_lifecycle_info(name)
        if lifecycle_meta:
            return self.lifecycle_manager._serialize_lifecycle_metadata(lifecycle_meta)
        return None
    
    def get_factors_needing_attention(self) -> List[str]:
        """Get factors that need attention based on lifecycle status"""
        needing_attention = []
        for uid, metadata in self._factors.items():
            if hasattr(metadata, "needs_attention") and metadata.needs_attention():
                needing_attention.append(metadata.name)
        return needing_attention
    
    def monitor_factor_performance(self, name: str) -> Dict:
        """Monitor factor performance and update lifecycle accordingly"""
        if not self.lifecycle_manager:
            return {}
        
        return self.lifecycle_manager._update_performance_metrics(name) or {}
    
    def bulk_status_update(self, factor_names: List[str],
                          new_status: FactorLifecycleStatus,
                          reason: str = "", operator: str = "system") -> Dict[str, bool]:
        """Update lifecycle status for multiple factors"""
        if not self.lifecycle_manager:
            raise RuntimeError("Lifecycle management is not enabled")
        
        results = {}
        for name in factor_names:
            try:
                results[name] = self.transition_factor_status(name, new_status, reason, operator)
            except Exception as e:
                results[name] = False
                self._logger.error(f"Failed to update status for factor '{name}': {e}")
        
        return results
    
    def refresh_from_database(self):
        """Refresh factor metadata from database"""
        if self._handler:
            self._load_from_database()
        else:
            self._logger.warning("No database handler available for refresh")
    
    def get_factor_dependencies_status(self, name: str) -> Dict:
        """Get the lifecycle status of all dependencies for a factor"""
        # First get the factor UID by name
        factor_uid = self.uid_manager.get_uid_by_name(name)
        if not factor_uid or factor_uid not in self._factors:
            raise ValueError(f"Factor '{name}' not registered")
        
        metadata = self._factors[factor_uid]
        dependencies_status = {}
        
        for dep_uid in metadata.dependencies:
            # Check if dependency is loaded in memory
            if dep_uid in self._factors:
                dep_metadata = self._factors[dep_uid]
                dep_name = dep_metadata.name
                dependencies_status[dep_name] = {
                    "status": getattr(dep_metadata, "lifecycle_status", FactorLifecycleStatus.DEVELOPMENT).value,
                    "version": str(getattr(dep_metadata, "version", FactorVersion())),
                    "production_ready": getattr(dep_metadata, "is_production_ready", lambda: False)()
                }
            else:
                # Dependency not in memory, mark as not found
                dependencies_status[dep_uid] = {
                    "status": "not_found",
                    "version": "unknown",
                    "production_ready": False
                }
        
        return dependencies_status
    
    def validate_factor_pipeline(self, factor_names: List[str]) -> Dict:
        """Validate that a factor pipeline is ready for production"""
        execution_order = self.get_execution_order(factor_names)
        validation_results = {
            "overall_status": "valid",
            "factors": {},
            "issues": []
        }
        
        for factor_uid in execution_order:
            if factor_uid not in self._factors:
                validation_results["issues"].append(f"Factor UID '{factor_uid}' not registered")
                validation_results["overall_status"] = "invalid"
                continue
            
            metadata = self._factors[factor_uid]
            factor_name = metadata.name
            factor_status = {
                "lifecycle_status": getattr(metadata, "lifecycle_status", FactorLifecycleStatus.DEVELOPMENT).value,
                "production_ready": getattr(metadata, "is_production_ready", lambda: False)(),
                "needs_attention": getattr(metadata, "needs_attention", lambda: False)(),
                "dependencies_ready": True
            }
            
            # Check dependencies
            deps_status = self.get_factor_dependencies_status(factor_name)
            for dep_name, dep_info in deps_status.items():
                if not dep_info["production_ready"]:
                    factor_status["dependencies_ready"] = False
                    validation_results["issues"].append(
                        f"Dependency '{dep_name}' of factor '{factor_name}' is not production ready"
                    )
            
            if not factor_status["production_ready"] or not factor_status["dependencies_ready"]:
                validation_results["overall_status"] = "warning"
            
            if factor_status["needs_attention"]:
                validation_results["overall_status"] = "critical"
                validation_results["issues"].append(f"Factor '{factor_name}' needs attention")
            
            validation_results["factors"][factor_name] = factor_status
        
        return validation_results


class FactorManager:
    """Main factor management class that integrates registry and storage"""
    
    def __init__(self, database_type: DatabaseTypes, enable_lifecycle: bool = True, uid_strategy: str = "uuid"):
        self.database_type = database_type
        self.enable_lifecycle = enable_lifecycle
        self.uid_strategy = uid_strategy
        
        # Create database handler
        self.handler = self._create_handler(database_type)
        
        # Initialize database
        self.handler.init_db()
        
        # Initialize registry with handler for persistence
        self.registry = FactorRegistry(
            handler=self.handler, 
            enable_lifecycle=enable_lifecycle,
            uid_strategy=uid_strategy
        )
        
        # Inherit registry's logger
        self._logger = self.registry._logger
    
    def __enter__(self):
        """Enter context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and cleanup resources"""
        self.close()
    
    def close(self):
        """Close database connections and cleanup resources"""
        if hasattr(self.handler, "close"):
            self.handler.close()
            self._logger.info("Database connection closed")
    
    def _create_handler(self, database_type: DatabaseTypes) -> FactorHandler:
        """Create appropriate factor handler based on database type"""
        if database_type == DatabaseTypes.MONGODB:
            return MongoFactorHandler()
        elif database_type == DatabaseTypes.DUCKDB:
            return DuckDBFactorHandler()
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    def register_factor(self, name: str, description: str = "",
                       dependencies: List[str] = None,
                       category: str = "default",
                       tags: List[str] = None,
                       author: str = "",
                       initial_status: FactorLifecycleStatus = FactorLifecycleStatus.DEVELOPMENT) -> str:
        """Register a factor with metadata and return its unique ID"""
        return self.registry.register_factor(
            name=name,
            description=description,
            dependencies=dependencies,
            category=category,
            tags=tags,
            author=author,
            initial_status=initial_status
        )
    
    def store_factors(self, factors: Dict[str, Factor], 
                     table_name: str,
                     factor_names: List[str] = None,
                     force_recompute: bool = False) -> None:
        """Store pre-computed Factor objects directly"""
        if factor_names is None:
            factor_names = list(factors.keys())
        
        execution_order = self.registry.get_execution_order(factor_names)
        
        for factor_uid in execution_order:
            # Convert UID to factor name for lookup in factors dict
            factor_name = self.registry.get_factor_name_by_uid(factor_uid)
            if not factor_name:
                self._logger.warning(f"Could not find factor name for UID '{factor_uid}'")
                continue
                
            if factor_name not in factors:
                self._logger.warning(f"No Factor object provided for factor '{factor_name}' (UID: {factor_uid})")
                continue
            
            factor_obj = factors[factor_name]
            if not isinstance(factor_obj, Factor):
                raise ValueError(f"Object for '{factor_name}' must be a Factor instance, got {type(factor_obj)}")
            
            # Check if factor already exists
            needs_storage = force_recompute or not self.handler.check_factor_exists(table_name, factor_name)
            
            if needs_storage:
                self._logger.info(f"Storing factor '{factor_name}'")
                try:
                    self.handler.factor2db(factor_obj, factor_name, table_name)
                    self._logger.info(f"Factor '{factor_name}' stored successfully")
                except Exception as e:
                    self._logger.error(f"Error storing factor '{factor_name}': {e}")
                    raise
            else:
                self._logger.info(f"Factor '{factor_name}' already exists, skipping storage")

    def store_string_factors(self, factor_strings: Dict[str, str], 
                           table_name: str,
                           additional_vars: dict = None,
                           factor_names: List[str] = None,
                           force_recompute: bool = False) -> None:
        """Store factors defined by string expressions"""
        from ..expression.utils.tools import calc_str
        
        if factor_names is None:
            factor_names = list(factor_strings.keys())
        
        execution_order = self.registry.get_execution_order(factor_names)
        computed_factors = {}
        
        for factor_uid in execution_order:
            # Convert UID to factor name for lookup in factor_strings dict
            factor_name = self.registry.get_factor_name_by_uid(factor_uid)
            if not factor_name:
                self._logger.warning(f"Could not find factor name for UID '{factor_uid}'")
                continue
                
            if factor_name not in factor_strings:
                self._logger.warning(f"No string expression provided for factor '{factor_name}' (UID: {factor_uid})")
                continue
            
            # Check if factor already exists
            needs_computation = force_recompute or not self.handler.check_factor_exists(table_name, factor_name)
            
            if needs_computation:
                self._logger.info(f"Computing factor '{factor_name}' from string expression")
                
                # Prepare additional variables including dependencies
                metadata = self.registry.get_factor_info(factor_name)
                merged_vars = additional_vars.copy() if additional_vars else {}
                
                # Add computed dependencies
                for dep_uid in metadata.dependencies:
                    dep_name = self.registry.get_factor_name_by_uid(dep_uid)

                    if dep_name in computed_factors:
                        merged_vars[dep_name] = computed_factors[dep_name]
                    else:
                        # Load from database
                        dep_factor = self.handler.read_factor(table_name, dep_name)
                        merged_vars[dep_name] = dep_factor
                
                try:
                    factor_obj = calc_str(factor_strings[factor_name], merged_vars)
                    if not isinstance(factor_obj, Factor):
                        raise ValueError(f"String expression for '{factor_name}' must evaluate to a Factor object")
                    
                    # Store to database
                    self.handler.factor2db(factor_obj, factor_name, table_name)
                    computed_factors[factor_name] = factor_obj
                    
                    self._logger.info(f"Factor '{factor_name}' computed from string and stored successfully")
                    
                except Exception as e:
                    self._logger.error(f"Error computing factor '{factor_name}' from string: {e}")
                    raise
            else:
                self._logger.info(f"Factor '{factor_name}' already exists, skipping computation")

    def compute_and_store_factors(self, factor_functions: Dict[str, callable], 
                                 table_name: str,
                                 factor_names: List[str] = None,
                                 force_recompute: bool = False,
                                 backfill_days: int = None) -> None:
        """Compute and store factors in dependency order"""
        if factor_names is None:
            factor_names = list(factor_functions.keys())
        
        execution_order = self.registry.get_execution_order(factor_names)
        
        computed_factors = {}
        
        for factor_uid in execution_order:
            # Convert UID to factor name for lookup in factor_functions dict
            factor_name = self.registry.get_factor_name_by_uid(factor_uid)
            if not factor_name:
                self._logger.warning(f"Could not find factor name for UID '{factor_uid}'")
                continue
                
            if factor_name not in factor_functions:
                self._logger.warning(f"No computation function provided for factor '{factor_name}' (UID: {factor_uid})")
                continue
            
            # Check if factor already exists and handle backfill
            needs_computation = force_recompute
            
            if not needs_computation:
                if self.handler.check_factor_exists(table_name, factor_name):
                    if backfill_days:
                        # Check if we need to backfill
                        start_date, end_date = self.handler.get_factor_date_range(table_name, factor_name)
                        target_start_date = pd.Timestamp.now() - pd.Timedelta(days=backfill_days)
                        if pd.Timestamp(start_date) > target_start_date:
                            needs_computation = True
                            self._logger.info(f"Backfilling factor '{factor_name}' from {target_start_date}")
                else:
                    needs_computation = True
            
            if needs_computation:
                self._logger.info(f"Computing factor '{factor_name}'")
                
                # Prepare dependencies
                metadata = self.registry.get_factor_info(factor_name)
                dep_data = {}
                for dep_uid in metadata.dependencies:
                    dep_name = self.registry.get_factor_name_by_uid(dep_uid)

                    if dep_name in computed_factors:
                        dep_data[dep_name] = computed_factors[dep_name]
                    else:
                        # Load from database
                        dep_data[dep_name] = self.handler.read_factor(table_name, dep_name)
                
                # Compute factor
                try:
                    factor_data = factor_functions[factor_name](**dep_data)
                    if not isinstance(factor_data, Factor):
                        raise ValueError(f"Factor function for '{factor_name}' must return a Factor object")
                    
                    # Store to database
                    self.handler.factor2db(factor_data, factor_name, table_name)
                    computed_factors[factor_name] = factor_data
                    
                    self._logger.info(f"Factor '{factor_name}' computed and stored successfully")
                    
                except Exception as e:
                    self._logger.error(f"Error computing factor '{factor_name}': {e}")
                    raise
            else:
                self._logger.info(f"Factor '{factor_name}' already exists, skipping computation")
    
    def get_factors(self, table_name: str, factor_names: List[str], 
                   start_date: str = None, end_date: str = None) -> Dict[str, Factor]:
        """Get multiple factors from database"""
        factors = {}
        for factor_name in factor_names:
            factors[factor_name] = self.handler.read_factor(
                table_name, factor_name, start_date, end_date
            )
        return factors
    
    def smart_backfill(self, table_name: str, target_days: int = 252,
                      factor_names: List[str] = None) -> None:
        """Intelligently backfill historical data"""
        if factor_names is None:
            factor_names = self.registry.list_factors(return_uids=False)
        
        target_start_date = pd.Timestamp.now() - pd.Timedelta(days=target_days)
        
        factors_to_backfill = []
        for factor_name in factor_names:
            if self.handler.check_factor_exists(table_name, factor_name):
                start_date, end_date = self.handler.get_factor_date_range(table_name, factor_name)
                if pd.Timestamp(start_date) > target_start_date:
                    factors_to_backfill.append(factor_name)
            else:
                factors_to_backfill.append(factor_name)
        
        if factors_to_backfill:
            self._logger.info(f"Backfilling {len(factors_to_backfill)} factors: {factors_to_backfill}")
            # This would need factor computation functions - placeholder for now
            # self.compute_and_store_factors(factor_functions, table_name, 
            #                               factors_to_backfill, backfill_days=target_days)
        else:
            self._logger.info("No factors need backfilling")
    
    # Lifecycle management methods
    def transition_factor_status(self, name: str, new_status: FactorLifecycleStatus,
                               reason: str = "", operator: str = "system") -> bool:
        """Transition factor lifecycle status"""
        return self.registry.transition_factor_status(name, new_status, reason, operator)
    
    def create_factor_version(self, name: str, version_type: str = "patch",
                            source_code: str = "", metadata: Dict = None,
                            operator: str = "system") -> FactorVersion:
        """Create a new version of a factor"""
        return self.registry.create_factor_version(name, version_type, source_code, metadata, operator)
    
    def get_lifecycle_dashboard(self) -> Dict:
        """Get a comprehensive lifecycle dashboard"""
        dashboard = {
            "summary": self.registry.get_graph_statistics(),
            "factors_by_status": {},
            "factors_needing_attention": self.registry.get_factors_needing_attention(),
            "recent_transitions": []
        }
        
        # Group factors by status
        for status in FactorLifecycleStatus:
            factors = self.registry.list_factors(status=status)
            dashboard["factors_by_status"][status.value] = factors
        
        return dashboard
    
    def automated_lifecycle_check(self) -> Dict:
        """Perform automated lifecycle checks and transitions"""
        if not hasattr(self.registry, "enable_lifecycle") or not self.registry.enable_lifecycle:
            return {"error": "Lifecycle management is not enabled"}
        
        results = {
            "performance_checks": [],
            "status_transitions": [],
            "warnings": []
        }
        
        # Check all active factors
        active_factors = self.registry.list_factors(
            status=FactorLifecycleStatus.ACTIVE
        )
        
        for factor_name in active_factors:
            try:
                perf_result = self.registry.monitor_factor_performance(factor_name)
                results["performance_checks"].append({
                    "factor": factor_name,
                    "result": perf_result
                })
            except Exception as e:
                results["warnings"].append({
                    "factor": factor_name,
                    "warning": f"Performance check failed: {e}"
                })
        
        return results
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def export_factor_portfolio(self, filename: str = None) -> Dict:
        """Export comprehensive factor portfolio information"""
        portfolio_data = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "total_factors": len(self.registry._factors),
            "statistics": self.registry.get_graph_statistics(),
            "factors": {}
        }
        
        # Export each factor's complete information
        for uid, metadata in self.registry._factors.items():
            factor_data = {
                "basic_info": {
                    "name": metadata.name,
                    "uid": uid,
                    "description": metadata.description,
                    "category": metadata.category,
                    "tags": metadata.tags,
                    "author": getattr(metadata, "author", ""),
                    "created_time": metadata.created_time.isoformat() if metadata.created_time else None,
                    "updated_time": metadata.updated_time.isoformat() if metadata.updated_time else None
                },
                "lifecycle": {
                    "status": getattr(metadata, "lifecycle_status", FactorLifecycleStatus.DEVELOPMENT).value,
                    "version": str(getattr(metadata, "version", FactorVersion())),
                    "is_production_ready": getattr(metadata, "is_production_ready", lambda: False)(),
                    "needs_attention": getattr(metadata, "needs_attention", lambda: False)()
                },
                "dependencies": metadata.dependencies
            }
            
            # Add detailed lifecycle info if available
            if hasattr(self.registry, "lifecycle_manager") and self.registry.lifecycle_manager:
                lifecycle_info = self.registry.get_lifecycle_info(metadata.name)
                if lifecycle_info:
                    factor_data["detailed_lifecycle"] = lifecycle_info
            
            portfolio_data["factors"][metadata.name] = factor_data
        
        # Save to file if filename provided
        if filename:
            import json
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(portfolio_data, f, indent=2, ensure_ascii=False)
            self._logger.info(f"Factor portfolio exported to {filename}")
        
        return portfolio_data
