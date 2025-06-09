import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

class CheckDeprecated:
    """
    Provides functionality to check if HPO terms are deprecated and find their replacements.
    
    This class loads the HPO ontology and provides methods to check the deprecation status
    of terms and find their updated replacements.
    
    Attributes:
        data_path: Path to the HPO JSON file
        terms: Dictionary of all HPO terms indexed by their ID
        deprecated_terms: Dictionary of deprecated HPO terms with their replacements
    """
    
    def __init__(self, data_path: str = "data/ontology/hp.json"):
        """
        Initialize the CheckDeprecated instance.
        
        Args:
            data_path: Path to the HPO JSON file
        """
        self.data_path = data_path
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.deprecated_terms: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load the ontology
        self._load_ontology()
        
    def _load_ontology(self) -> None:
        """
        Load the HPO ontology from the JSON file and identify deprecated terms.
        """
        if not os.path.exists(self.data_path):
            self.logger.error(f"HPO ontology file not found at {self.data_path}")
            return
            
        try:
            self.logger.info(f"Loading HPO ontology from {self.data_path}")
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
            # Process the graph structure
            if "graphs" in data and isinstance(data["graphs"], list) and data["graphs"]:
                graph_data = data["graphs"][0]
                
                # Process all nodes
                for node in graph_data.get("nodes", []):
                    # Extract HPO ID and check if it's a valid HPO term
                    node_id = node.get("id", "")
                    if not node_id or not isinstance(node_id, str):
                        continue
                        
                    # Extract the HPO ID (format: http://purl.obolibrary.org/obo/HP_0000001)
                    if "/" in node_id:
                        hpo_id = node_id.split("/")[-1]
                    else:
                        hpo_id = node_id
                        
                    # Normalize to HP:NNNNNNN format
                    hpo_id = self._normalize_hpo_id(hpo_id)
                    
                    # Skip non-HPO terms
                    if not hpo_id.startswith("HP:"):
                        continue
                        
                    # Store the term
                    self.terms[hpo_id] = {
                        "id": hpo_id,
                        "label": node.get("lbl", ""),
                        "meta": node.get("meta", {})
                    }
                    
                    # Check if deprecated and store the replacement
                    meta = node.get("meta", {})
                    if meta and meta.get("deprecated", False):
                        # Look for the replacement term ID in basicPropertyValues
                        replacement_id = None
                        basic_property_values = meta.get("basicPropertyValues", [])
                        
                        for prop in basic_property_values:
                            if prop.get("pred") == "http://purl.obolibrary.org/obo/IAO_0100001":
                                replacement_value = prop.get("val", "")
                                # If the replacement is an HP ID, store it
                                if replacement_value.startswith("HP:"):
                                    replacement_id = replacement_value
                                    break
                                    
                        if replacement_id:
                            self.deprecated_terms[hpo_id] = replacement_id
                            self.logger.debug(f"Found deprecated term {hpo_id} with replacement {replacement_id}")
                        else:
                            self.logger.debug(f"Found deprecated term {hpo_id} without replacement")
            
            self.logger.info(f"Loaded {len(self.terms)} HPO terms, {len(self.deprecated_terms)} deprecated terms")
            
        except Exception as e:
            self.logger.error(f"Error loading HPO ontology: {str(e)}")
            
    def _normalize_hpo_id(self, hpo_id: str) -> str:
        """
        Normalize an HPO ID to the standard format (HP:NNNNNNN).
        
        Args:
            hpo_id: HPO ID in any format (e.g., HP_0000123, HP:0000123)
            
        Returns:
            Normalized HPO ID (HP:NNNNNNN)
        """
        # Replace underscore with colon if needed
        if "_" in hpo_id:
            hpo_id = hpo_id.replace("_", ":")
            
        # Ensure proper format
        if hpo_id.startswith("HP:"):
            return hpo_id
            
        # Handle edge cases
        if hpo_id.startswith("HP"):
            return f"HP:{hpo_id[2:]}"
            
        return hpo_id
            
    def is_deprecated(self, term_id: str) -> bool:
        """
        Check if an HPO term is deprecated.
        
        Args:
            term_id: HPO term ID to check
            
        Returns:
            True if the term is deprecated, False otherwise
        """
        # Normalize the term ID
        term_id = self._normalize_hpo_id(term_id)
        
        # Check if term exists and is deprecated
        if term_id in self.terms:
            meta = self.terms[term_id].get("meta", {})
            return meta.get("deprecated", False)
            
        return False
        
    def get_replacement(self, term_id: str) -> Optional[str]:
        """
        Get the replacement for a deprecated HPO term.
        
        Args:
            term_id: HPO term ID to get replacement for
            
        Returns:
            Replacement term ID if available, None otherwise
        """
        # Normalize the term ID
        term_id = self._normalize_hpo_id(term_id)
        
        # Return the replacement if term is deprecated and has a replacement
        if term_id in self.deprecated_terms:
            return self.deprecated_terms[term_id]
            
        return None
        
    def check_and_replace(self, term_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a term is deprecated and get its replacement in one operation.
        
        Args:
            term_id: HPO term ID to check
            
        Returns:
            Tuple of (is_deprecated, replacement_id)
        """
        is_deprecated = self.is_deprecated(term_id)
        replacement = self.get_replacement(term_id) if is_deprecated else None
        return is_deprecated, replacement 