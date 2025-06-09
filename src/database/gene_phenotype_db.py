from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Set
import csv
import os
import logging
from src.core.types import Phenotype

class GenePhenotypeDatabase(ABC):
    """Interface for accessing gene-phenotype relationship data."""
    
    @abstractmethod
    def get_phenotypes_for_gene(self, gene_symbol: str) -> List[Phenotype]:
        """Get all phenotypes associated with a given gene."""
        pass
    
    @abstractmethod
    def get_genes_for_phenotype(self, hpo_id: str) -> List[str]:
        """Get all genes associated with a given phenotype."""
        pass
    
    @abstractmethod
    def get_frequency(self, gene_symbol: str, hpo_id: str, disease_id: Optional[str] = None) -> str:
        """Get the frequency of a phenotype for a given gene/disease combination."""
        pass
    
    @abstractmethod
    def get_diseases_for_gene(self, gene_symbol: str) -> List[str]:
        """Get all diseases associated with a given gene."""
        pass

    @abstractmethod
    def get_genes_for_disease(self, disease_id: str) -> List[str]:
        """Get all genes associated with a given disease."""
        pass


class HPOAGenePhenotypeDatabase(GenePhenotypeDatabase):
    """Implementation of GenePhenotypeDatabase for HPOA format gene-phenotype data."""
    
    def __init__(self, genes_to_phenotypes_path: str, phenotype_to_genes_path: Optional[str] = None):
        """
        Initialize the database with paths to data files.
        
        Args:
            genes_to_phenotypes_path: Path to genes_to_phenotypes.txt
            phenotype_to_genes_path: Path to phenotype_to_genes.txt (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.genes_to_phenotypes_path = genes_to_phenotypes_path
        self.phenotype_to_genes_path = phenotype_to_genes_path
        
        # Internal data structures
        self.gene_to_phenotypes: Dict[str, List[Dict[str, Any]]] = {}
        self.phenotype_to_genes: Dict[str, List[Dict[str, Any]]] = {}
        self.gene_disease_phenotypes: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.disease_to_genes: Dict[str, Set[str]] = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data from the provided file paths."""
        if not os.path.exists(self.genes_to_phenotypes_path):
            raise FileNotFoundError(f"Gene-phenotype file not found: {self.genes_to_phenotypes_path}")
        
        # Load genes_to_phenotypes.txt
        with open(self.genes_to_phenotypes_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 6:
                    self.logger.warning(f"Skipping malformed row: {row}")
                    continue
                
                gene_id, gene_symbol, hpo_id, hpo_name, frequency, disease_id = row[:6]
                
                # Add to gene_to_phenotypes
                if gene_symbol not in self.gene_to_phenotypes:
                    self.gene_to_phenotypes[gene_symbol] = []
                
                phenotype_info = {
                    "ncbi_gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                    "hpo_id": hpo_id,
                    "hpo_name": hpo_name,
                    "frequency": frequency,
                    "disease_id": disease_id
                }
                
                self.gene_to_phenotypes[gene_symbol].append(phenotype_info)
                
                # Add to gene_disease_phenotypes
                key = (gene_symbol, disease_id)
                if key not in self.gene_disease_phenotypes:
                    self.gene_disease_phenotypes[key] = []
                self.gene_disease_phenotypes[key].append(phenotype_info)
                
                # Add to disease_to_genes
                if disease_id not in self.disease_to_genes:
                    self.disease_to_genes[disease_id] = set()
                self.disease_to_genes[disease_id].add(gene_symbol)
        
        # Load phenotype_to_genes.txt if provided
        if self.phenotype_to_genes_path and os.path.exists(self.phenotype_to_genes_path):
            with open(self.phenotype_to_genes_path, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                headers = next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) < 5:
                        self.logger.warning(f"Skipping malformed row in phenotype_to_genes: {row}")
                        continue
                    
                    hpo_id, hpo_name, gene_id, gene_symbol, disease_id = row[:5]
                    
                    # Add to phenotype_to_genes
                    if hpo_id not in self.phenotype_to_genes:
                        self.phenotype_to_genes[hpo_id] = []
                    
                    gene_info = {
                        "hpo_id": hpo_id,
                        "hpo_name": hpo_name,
                        "ncbi_gene_id": gene_id,
                        "gene_symbol": gene_symbol,
                        "disease_id": disease_id
                    }
                    
                    self.phenotype_to_genes[hpo_id].append(gene_info)
    
    def get_phenotypes_for_gene(self, gene_symbol: str) -> List[Phenotype]:
        """
        Get all phenotypes associated with a given gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "AARS1")
            
        Returns:
            List of Phenotype objects associated with the gene
        """
        if gene_symbol not in self.gene_to_phenotypes:
            return []
        
        phenotypes = []
        for phenotype_info in self.gene_to_phenotypes[gene_symbol]:
            metadata = {
                "gene_symbol": phenotype_info["gene_symbol"],
                "ncbi_gene_id": phenotype_info["ncbi_gene_id"],
                "frequency": phenotype_info["frequency"],
                "disease_id": phenotype_info["disease_id"]
            }
            
            phenotype = Phenotype(
                id=phenotype_info["hpo_id"],
                name=phenotype_info["hpo_name"],
                metadata=metadata
            )
            
            phenotypes.append(phenotype)
        
        return phenotypes
    
    def get_genes_for_phenotype(self, hpo_id: str) -> List[str]:
        """
        Get all genes associated with a given phenotype.
        
        Args:
            hpo_id: HPO ID (e.g., "HP:0001939")
            
        Returns:
            List of gene symbols associated with the phenotype
        """
        genes = set()
        
        # First try phenotype_to_genes if available
        if self.phenotype_to_genes and hpo_id in self.phenotype_to_genes:
            for gene_info in self.phenotype_to_genes[hpo_id]:
                genes.add(gene_info["gene_symbol"])
            return list(genes)
        
        # Otherwise, scan through gene_to_phenotypes
        for gene_symbol, phenotype_infos in self.gene_to_phenotypes.items():
            for phenotype_info in phenotype_infos:
                if phenotype_info["hpo_id"] == hpo_id:
                    genes.add(gene_symbol)
        
        return list(genes)
    
    def get_frequency(self, gene_symbol: str, hpo_id: str, disease_id: Optional[str] = None) -> str:
        """
        Get the frequency of a phenotype for a given gene/disease combination.
        
        Args:
            gene_symbol: Gene symbol (e.g., "AARS1")
            hpo_id: HPO ID (e.g., "HP:0001939")
            disease_id: Optional disease ID to filter by (e.g., "OMIM:243400")
            
        Returns:
            Frequency string if found, empty string otherwise
        """
        if gene_symbol not in self.gene_to_phenotypes:
            return ""
        
        for phenotype_info in self.gene_to_phenotypes[gene_symbol]:
            if phenotype_info["hpo_id"] == hpo_id:
                if disease_id is None or phenotype_info["disease_id"] == disease_id:
                    return phenotype_info["frequency"]
        
        return ""
    
    def get_diseases_for_gene(self, gene_symbol: str) -> List[str]:
        """
        Get all diseases associated with a given gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "AARS1")
            
        Returns:
            List of disease IDs associated with the gene
        """
        if gene_symbol not in self.gene_to_phenotypes:
            return []
        
        diseases = set()
        for phenotype_info in self.gene_to_phenotypes[gene_symbol]:
            if phenotype_info["disease_id"]:
                diseases.add(phenotype_info["disease_id"])
        
        return list(diseases)
    
    def get_genes_for_disease(self, disease_id: str) -> List[str]:
        """
        Get all genes associated with a given disease.
        
        Args:
            disease_id: Disease ID (e.g., "OMIM:243400")
            
        Returns:
            List of gene symbols associated with the disease
        """
        if disease_id in self.disease_to_genes:
            return list(self.disease_to_genes[disease_id])
        return [] 