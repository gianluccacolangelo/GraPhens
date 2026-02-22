import xml.etree.ElementTree as ET
import logging
import os
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional, Tuple

class OrphanetMapper:
    """
    Maps genes to Orphanet Disease Categories using ORDO and HPO gene-disease associations.
    """
    
    NS = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'owl': 'http://www.w3.org/2002/07/owl#',
        'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#'
    }

    # The root node for "Rare Disease" classification in ORDO is often considered
    # "Rare disorder" ORPHA:377788 or similar.
    # Let's verify the root. Often ORPHA:377788 is "Disease". 
    # Children of ORPHA:377788 are typically the chapters like "Rare neurologic disease".
    DISEASE_ROOT_ID = 'ORPHA:377788' 

    def __init__(self, ordo_path: str, gene_phenotype_path: str, root_categories: Optional[Dict[str, str]] = None):
        self.logger = logging.getLogger(__name__)
        self.ordo_path = ordo_path
        self.gene_phenotype_path = gene_phenotype_path
        
        self.gene_to_diseases: Dict[str, Set[str]] = defaultdict(set)
        self.omim_to_orpha: Dict[str, str] = {}
        self.orpha_parents: Dict[str, Set[str]] = defaultdict(set)
        self.orpha_labels: Dict[str, str] = {}
        self.root_categories: Dict[str, str] = {} # ID -> Label
        self.custom_roots = root_categories
        
        self._load_gene_map()
        self._load_ontology()
        
        if self.custom_roots:
            self.root_categories = self.custom_roots
            self.logger.info(f"Using {len(self.root_categories)} custom root categories")
        else:
            self._identify_root_categories()

    def _load_gene_map(self):
        """Parses genes_to_phenotype.txt to map Gene Symbol -> Disease IDs (OMIM/ORPHA)"""
        self.logger.info(f"Loading gene-disease map from {self.gene_phenotype_path}")
        try:
            with open(self.gene_phenotype_path, 'r') as f:
                # Skip header
                next(f) 
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        gene_symbol = parts[1]
                        disease_id = parts[5] # OMIM:1234 or ORPHA:1234
                        self.gene_to_diseases[gene_symbol].add(disease_id)
        except Exception as e:
            self.logger.error(f"Error loading genes_to_phenotype: {e}")

    def _load_ontology(self):
        """Parses ORDO OWL file to extract hierarchy and mappings"""
        self.logger.info(f"Loading ORDO ontology from {self.ordo_path}")
        try:
            # Register namespaces
            for prefix, uri in self.NS.items():
                ET.register_namespace(prefix, uri)
                
            tree = ET.parse(self.ordo_path)
            root = tree.getroot()
            
            # Iterate over all Classes
            for cls in root.findall('owl:Class', self.NS):
                rdf_about = cls.get(f"{{{self.NS['rdf']}}}about")
                if not rdf_about or 'Orphanet_' not in rdf_about:
                    continue
                
                orpha_id = 'ORPHA:' + rdf_about.split('Orphanet_')[-1]
                
                # Get Label
                label_node = cls.find('rdfs:label', self.NS)
                if label_node is not None:
                    self.orpha_labels[orpha_id] = label_node.text
                
                # Get Parents (subClassOf)
                for sub in cls.findall('rdfs:subClassOf', self.NS):
                    res = sub.get(f"{{{self.NS['rdf']}}}resource")
                    if res and 'Orphanet_' in res:
                        parent_id = 'ORPHA:' + res.split('Orphanet_')[-1]
                        self.orpha_parents[orpha_id].add(parent_id)
                        
                # Get OMIM Mappings (hasDbXref)
                for xref in cls.findall('oboInOwl:hasDbXref', self.NS):
                    if xref.text and xref.text.startswith('OMIM:'):
                        self.omim_to_orpha[xref.text] = orpha_id
                        
        except Exception as e:
            self.logger.error(f"Error loading ORDO ontology: {e}")

    def _identify_root_categories(self):
        """Identifies the direct children of the Disease Root as categories"""
        # Find children of DISEASE_ROOT_ID
        # Since we stored child->parent, we iterate all nodes to find who has ROOT as parent
        for child, parents in self.orpha_parents.items():
            if self.DISEASE_ROOT_ID in parents:
                label = self.orpha_labels.get(child, child)
                self.root_categories[child] = label
        
        self.logger.info(f"Identified {len(self.root_categories)} root categories in ORDO")

    def get_category_for_gene(self, gene_symbol: str) -> str:
        """Returns the dominant Orphanet category for a gene"""
        diseases = self.gene_to_diseases.get(gene_symbol, set())
        if not diseases:
            return "Unknown"
            
        categories = []
        for disease_id in diseases:
            orpha_id = None
            
            # Resolve to ORPHA ID
            if disease_id.startswith('ORPHA:'):
                orpha_id = disease_id
            elif disease_id.startswith('OMIM:'):
                orpha_id = self.omim_to_orpha.get(disease_id)
            
            if not orpha_id:
                continue
                
            # Traverse up to find category
            cat = self._find_ancestor_category(orpha_id)
            if cat:
                categories.append(cat)
        
        if not categories:
            return "Uncategorized"
            
        # Return most common category
        return Counter(categories).most_common(1)[0][0]

    def _find_ancestor_category(self, orpha_id: str) -> Optional[str]:
        """BFS up the hierarchy to find a root category"""
        queue = [orpha_id]
        visited = {orpha_id}
        
        while queue:
            current = queue.pop(0)
            
            if current in self.root_categories:
                return self.root_categories[current]
                
            for parent in self.orpha_parents.get(current, []):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        
        return None

