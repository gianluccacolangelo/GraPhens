import os
import json
import logging
import networkx as nx
from typing import Dict, Set, Any, Optional
from datetime import datetime

# Global cache for HPO provider instances
_hpo_provider_cache = {}

class HPOGraphProvider:
    """
    Provides access to the Human Phenotype Ontology (HPO) graph for efficient traversal.
    
    This class loads the HPO ontology from a JSON file and provides methods for
    traversing the graph to find ancestors, descendants, and term metadata.
    
    Attributes:
        graph: A NetworkX DiGraph representing the HPO ontology
        terms: Dictionary mapping HPO term IDs to their metadata
        version: Version information about the loaded HPO data
        last_loaded: Timestamp when the data was last loaded
    """
    
    def __init__(self, data_dir: str = "data/ontology"):
        """
        Initialize the HPO graph provider.
        
        Args:
            data_dir: Directory where HPO data files are stored
        """
        self.data_dir = data_dir
        self.graph = nx.DiGraph()
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.version: Optional[str] = None
        self.last_loaded: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Default filenames
        self.json_file = os.path.join(data_dir, "hp.json")
        self.obo_file = os.path.join(data_dir, "hp.obo")
        
    @classmethod
    def get_instance(cls, data_dir: str = "data/ontology") -> 'HPOGraphProvider':
        """
        Get a cached instance of HPOGraphProvider or create a new one.
        
        This class method provides a way to reuse HPO graph providers across
        different components, avoiding redundant loading of the ontology.
        
        Args:
            data_dir: Directory where HPO data files are stored
            
        Returns:
            An instance of HPOGraphProvider for the specified data directory
        """
        # Create a unique key for this configuration
        key = f"hpo_provider_{data_dir}"
        
        # Return cached instance if it exists and is loaded
        if key in _hpo_provider_cache:
            provider = _hpo_provider_cache[key]
            if provider.last_loaded is not None:
                return provider
                
        # Create a new instance if not in cache
        provider = cls(data_dir=data_dir)
        _hpo_provider_cache[key] = provider
        
        # Load the ontology
        provider.load()
        
        return provider
        
    def load(self, force_reload: bool = False) -> bool:
        """
        Load the HPO ontology from the data files.
        
        Args:
            force_reload: Whether to force reload even if already loaded
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not force_reload and self.last_loaded is not None:
            return True
            
        # Try to load from JSON first (faster)
        if os.path.exists(self.json_file):
            success = self._load_from_json()
        # Fall back to OBO if JSON is not available
        elif os.path.exists(self.obo_file):
            success = self._load_from_obo()
        else:
            self.logger.error(f"No HPO data files found in {self.data_dir}")
            return False
            
        if success:
            self.last_loaded = datetime.now()
            self.logger.info(f"Loaded HPO ontology version {self.version}")
            return True
        return False
    
    def _load_from_json(self) -> bool:
        """
        Load the HPO ontology from a JSON file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading HPO from JSON: {self.json_file}")
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                
            # Clear existing data
            self.graph.clear()
            self.terms = {}
            
            # Handle different JSON structures
            if "graphs" in data:
                # New JSON structure with graphs, nodes, and edges
                return self._load_from_json_graph_structure(data)
            elif "terms" in data:
                # Legacy JSON structure with terms
                return self._load_from_json_terms_structure(data)
            else:
                self.logger.error("Unknown JSON structure, could not parse HPO data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading HPO from JSON: {str(e)}")
            return False
    
    def _load_from_json_graph_structure(self, data: Dict) -> bool:
        """
        Load the HPO ontology from a JSON file with graphs, nodes, and edges structure.
        
        Args:
            data: The parsed JSON data
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Get the first graph (usually there's only one)
            if not data.get("graphs"):
                self.logger.error("No graphs found in JSON data")
                return False
                
            graph_data = data["graphs"][0]
            
            # Extract version info (from meta if available)
            meta = graph_data.get("meta", {})
            self.version = meta.get("version", None)
            if not self.version and "date" in meta:
                self.version = meta.get("date", None)
                
            # Process nodes
            nodes_count = 0
            for node in graph_data.get("nodes", []):
                # Extract the HPO ID from the URI
                uri = node.get("id", "")
                if not uri or not isinstance(uri, str):
                    continue
                    
                # Try to extract ID - handle both http://purl.obolibrary.org/obo/HP_0000001 and HP_0000001 formats
                if "/" in uri:
                    term_id = uri.split("/")[-1]
                else:
                    term_id = uri
                
                # Normalize ID format - allow both HP_NNNNNNN and HP:NNNNNNN
                term_id = self._normalize_hpo_id(term_id)
                
                if not term_id.startswith("HP:"):
                    continue
                
                # Extract definition if available
                definition = None
                node_meta = node.get("meta", {})
                if node_meta and "definition" in node_meta:
                    definition_obj = node_meta.get("definition")
                    if isinstance(definition_obj, dict) and "val" in definition_obj:
                        definition = definition_obj.get("val")
                    elif isinstance(definition_obj, str):
                        definition = definition_obj
                
                # Create term metadata
                term_data = {
                    "id": term_id,
                    "name": node.get("lbl", ""),
                    "type": node.get("type", ""),
                    "definition": definition,
                    "meta": node_meta
                }
                
                # Store term metadata
                self.terms[term_id] = term_data
                
                # Add node to graph
                self.graph.add_node(term_id, **term_data)
                nodes_count += 1
            
            # Process edges - determine if edges point from child->parent or parent->child
            # This is important for understanding the correct traverse direction for ancestors/descendants
            # In most ontology representations, edges should point from child to parent for is_a relationships
            
            # First pass: check for the relationship type pattern
            edges_count = 0
            subclass_pred = None
            
            # Try to find the subClassOf predicate
            for edge in graph_data.get("edges", []):
                pred = edge.get("pred", "")
                if pred and isinstance(pred, str):
                    if "subClassOf" in pred.lower() or "is_a" in pred.lower():
                        subclass_pred = pred
                        break
            
            # Second pass: Add all edges
            for edge in graph_data.get("edges", []):
                # Extract source and target
                sub = edge.get("sub", "")
                obj = edge.get("obj", "")
                pred = edge.get("pred", "")
                
                if not sub or not obj or not isinstance(sub, str) or not isinstance(obj, str):
                    continue
                
                # Normalize ID formats
                source_id = self._normalize_hpo_id(sub.split("/")[-1] if "/" in sub else sub)
                target_id = self._normalize_hpo_id(obj.split("/")[-1] if "/" in obj else obj)
                
                # Skip non-HPO terms
                if not source_id.startswith("HP:") or not target_id.startswith("HP:"):
                    continue
                
                # Check if this is a subClassOf/is_a relationship
                if pred == subclass_pred or (pred and ("subClassOf" in pred.lower() or "is_a" in pred.lower())):
                    # In ontologies, edges for is_a/subClassOf should go from child to parent
                    # Add edge in the right direction (child -> parent)
                    self.graph.add_edge(source_id, target_id)
                    edges_count += 1
            
            self.logger.info(f"Loaded {nodes_count} HPO terms and {edges_count} relationships")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading graph structure from JSON: {str(e)}")
            return False
    
    def _load_from_json_terms_structure(self, data: Dict) -> bool:
        """
        Load the HPO ontology from a JSON file with terms structure.
        
        Args:
            data: The parsed JSON data
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Extract version info
        self.version = data.get('version', None)
        
        # Process nodes (terms)
        for term in data.get('terms', []):
            term_id = term.get('id')
            if not term_id:
                continue
                
            # Normalize ID format
            term_id = self._normalize_hpo_id(term_id)
                
            # Store term metadata
            self.terms[term_id] = term
            
            # Add node to graph
            self.graph.add_node(term_id, **term)
            
            # Process edges (parent-child relationships)
            for parent in term.get('is_a', []):
                if isinstance(parent, str):
                    parent_id = self._normalize_hpo_id(parent)
                    self.graph.add_edge(parent_id, term_id)
                elif isinstance(parent, dict) and 'id' in parent:
                    parent_id = self._normalize_hpo_id(parent['id'])
                    self.graph.add_edge(parent_id, term_id)
        
        self.logger.info(f"Loaded {len(self.terms)} HPO terms and {self.graph.number_of_edges()} relationships")
        return True
    
    def _load_from_obo(self) -> bool:
        """
        Load the HPO ontology from an OBO file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading HPO from OBO: {self.obo_file}")
            
            # Try to use pronto if available (better OBO parsing)
            try:
                import pronto
                return self._load_from_obo_pronto()
            except ImportError:
                # Fall back to manual parsing
                pass
                
            # Clear existing data
            self.graph.clear()
            self.terms = {}
            
            current_term = None
            with open(self.obo_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Extract version
                    if line.startswith('format-version:'):
                        self.version = line.split('format-version:')[1].strip()
                    
                    # Start of a term block
                    elif line == '[Term]':
                        current_term = {}
                    
                    # End of a term block
                    elif line == '' and current_term is not None and 'id' in current_term:
                        term_id = current_term['id']
                        # Normalize ID
                        term_id = self._normalize_hpo_id(term_id)
                        self.terms[term_id] = current_term
                        self.graph.add_node(term_id, **current_term)
                        
                        # Add edges for is_a relationships
                        for parent_id in current_term.get('is_a', []):
                            parent_id = self._normalize_hpo_id(parent_id)
                            self.graph.add_edge(parent_id, term_id)
                            
                        current_term = None
                    
                    # Process term attributes
                    elif current_term is not None and ':' in line:
                        key, value = [x.strip() for x in line.split(':', 1)]
                        
                        if key == 'id':
                            current_term['id'] = value
                        elif key == 'name':
                            current_term['name'] = value
                        elif key == 'def':
                            current_term['definition'] = value
                        elif key == 'is_a' and value:
                            # Extract the ID from the is_a value (format: "HP:1234567 ! Term name")
                            parent_id = value.split(' ! ')[0].strip()
                            if 'is_a' not in current_term:
                                current_term['is_a'] = []
                            current_term['is_a'].append(parent_id)
            
            self.logger.info(f"Loaded {len(self.terms)} HPO terms and {self.graph.number_of_edges()} relationships")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading HPO from OBO: {str(e)}")
            return False
            
    def _load_from_obo_pronto(self) -> bool:
        """
        Load the HPO ontology from an OBO file using the pronto library.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        import pronto
        
        # Clear existing data
        self.graph.clear()
        self.terms = {}
        
        # Load ontology
        ontology = pronto.Ontology(self.obo_file)
        self.version = getattr(ontology, 'meta', {}).get('format-version', None)
        
        # Process terms
        for term_id, term in ontology.terms.items():
            # Skip non-HPO terms
            if not term_id.startswith('HP:'):
                continue
                
            # Create term metadata
            term_data = {
                'id': term_id,
                'name': term.name,
                'definition': term.definition or None,
                'comment': term.comment or None,
                'synonyms': [str(s) for s in term.synonyms],
                'relationships': {}
            }
            
            # Store term metadata
            self.terms[term_id] = term_data
            self.graph.add_node(term_id, **term_data)
            
            # Add edges for parent-child relationships
            for parent in term.relationships.get('is_a', []):
                parent_id = parent.id
                if parent_id.startswith('HP:'):
                    self.graph.add_edge(parent_id, term_id)
        
        self.logger.info(f"Loaded {len(self.terms)} HPO terms and {self.graph.number_of_edges()} relationships")
        return True
    
    def _normalize_hpo_id(self, hpo_id: str) -> str:
        """
        Normalize HPO ID format to use colons (HP:NNNNNNN) instead of underscores.
        
        Args:
            hpo_id: HPO ID in any format
            
        Returns:
            Normalized HPO ID
        """
        # First, handle any URI prefixes
        if "/" in hpo_id:
            hpo_id = hpo_id.split("/")[-1]
            
        # Handle HP_ vs HP: format
        if hpo_id.startswith("HP_"):
            return hpo_id.replace("HP_", "HP:", 1)
        
        # Already in HP: format or other format
        return hpo_id
    
    def get_ancestors(self, term_id: str) -> Set[str]:
        """
        Get all ancestors (both direct parents and indirect ancestors) of a given HPO term.
        
        This method performs a breadth-first traversal up the HPO hierarchy starting from
        the given term, collecting all terms encountered along the way. This includes both:
        - Direct parents (immediate ancestors)
        - Indirect ancestors (parents of parents, etc. all the way to the root terms)
        
        Args:
            term_id: HPO term ID
            
        Returns:
            Set of HPO term IDs representing all ancestors (direct and indirect)
        """
        if not self.load():
            return set()
        
        # Normalize the input ID
        term_id = self._normalize_hpo_id(term_id)
            
        if term_id not in self.graph:
            self.logger.warning(f"Term ID {term_id} not found in HPO graph")
            return set()
            
        # Get all ancestors by traversing from the term to the root(s)
        ancestors = set()
        queue = [term_id]
        visited = set(queue)
        
        while queue:
            current = queue.pop(0)
            for parent in self.graph.successors(current):  # Changed from predecessors to successors since edge direction is child->parent
                if parent not in visited:
                    ancestors.add(parent)
                    queue.append(parent)
                    visited.add(parent)
                    
        return ancestors

    def get_direct_parents(self, term_id: str) -> Set[str]:
        """
        Get the direct parents (immediate ancestors) of a given HPO term.

        Args:
            term_id: HPO term ID

        Returns:
            Set of HPO term IDs representing the direct parents.
        """
        if not self.load():
            return set()

        # Normalize the input ID
        term_id = self._normalize_hpo_id(term_id)

        if term_id not in self.graph:
            self.logger.warning(f"Term ID {term_id} not found in HPO graph")
            return set()

        # Since edges are child -> parent, successors are the direct parents
        parents = set(self.graph.successors(term_id))
        return parents

    def get_direct_children(self, term_id: str) -> Set[str]:
        """
        Get the direct children (immediate descendants) of a given HPO term.

        Args:
            term_id: HPO term ID

        Returns:
            Set of HPO term IDs representing the direct children.
        """
        if not self.load():
            return set()

        # Normalize the input ID
        term_id = self._normalize_hpo_id(term_id)

        if term_id not in self.graph:
            self.logger.warning(f"Term ID {term_id} not found in HPO graph")
            return set()

        # Since edges are child -> parent, predecessors are the direct children
        children = set(self.graph.predecessors(term_id))
        return children

    def get_descendants(self, term_id: str) -> Set[str]:
        """
        Get all descendants of a given HPO term.
        
        Args:
            term_id: HPO term ID
            
        Returns:
            Set of HPO term IDs representing all descendants
        """
        if not self.load():
            return set()
        
        # Normalize the input ID
        term_id = self._normalize_hpo_id(term_id)
            
        if term_id not in self.graph:
            self.logger.warning(f"Term ID {term_id} not found in HPO graph")
            return set()
            
        # Get all descendants by traversing from the term to the leaves
        descendants = set()
        queue = [term_id]
        visited = set(queue)
        
        while queue:
            current = queue.pop(0)
            for child in self.graph.predecessors(current):  # Changed from successors to predecessors since edge direction is child->parent
                if child not in visited:
                    descendants.add(child)
                    queue.append(child)
                    visited.add(child)
                    
        return descendants
        
    def get_metadata(self, term_id: str) -> Dict[str, Any]:
        """
        Get metadata for a given HPO term.
        
        Args:
            term_id: HPO term ID
            
        Returns:
            Dictionary containing term metadata
        """
        if not self.load():
            return {}
        
        # Normalize the input ID
        term_id = self._normalize_hpo_id(term_id)
            
        return self.terms.get(term_id, {}) 
