from typing import List, Dict, Optional, Set, Union
from src.core.types import Phenotype
from src.simulation.gene_phenotype.database import GenePhenotypeDatabase


class GenePhenotypeFacade:
    """
    A facade providing simplified access to gene-phenotype data.
    
    This class provides a clean interface for accessing gene-phenotype relationships
    and integrates well with other GraPhens components.
    """
    
    def __init__(self, data_dir: str = "data/gene_phenotype"):
        """
        Initialize the gene-phenotype facade.
        
        Args:
            data_dir: Directory containing the gene-phenotype database files
        """
        self.database = GenePhenotypeDatabase(data_dir=data_dir)
        
    def get_phenotypes_for_gene(self, gene: str) -> List[Phenotype]:
        """
        Get all phenotypes associated with a gene.
        
        Args:
            gene: Gene symbol (e.g., 'AARS1')
            
        Returns:
            List of Phenotype objects associated with the gene
        """
        return self.database.get_phenotypes_for_gene(gene)
    
    def get_genes_for_phenotype(self, phenotype_id: str, 
                               include_ancestors: bool = False) -> List[Dict[str, str]]:
        """
        Get all genes associated with a phenotype.
        
        Args:
            phenotype_id: HPO term ID (e.g., 'HP:0001939')
            include_ancestors: Whether to include genes associated with ancestor terms
            
        Returns:
            List of gene info dictionaries
        """
        return self.database.get_genes_for_phenotype(phenotype_id, include_ancestors)
    
    def get_frequency_information(self, gene: str, 
                                 phenotype_id: Optional[str] = None,
                                 disease_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get frequency information for phenotypes associated with a gene/disease.
        
        Args:
            gene: Gene symbol (e.g., 'AARS1')
            phenotype_id: Optional HPO term ID to filter by specific phenotype
            disease_id: Optional disease ID to filter by specific disease
            
        Returns:
            List of dictionaries with frequency information
        """
        return self.database.get_frequency_data(gene, phenotype_id, disease_id)
    
    def get_diseases_for_gene(self, gene: str) -> List[str]:
        """
        Get all diseases associated with a gene.
        
        Args:
            gene: Gene symbol (e.g., 'AARS1')
            
        Returns:
            List of disease IDs
        """
        return self.database.get_diseases_for_gene(gene)
    
    def get_available_genes(self) -> List[str]:
        """
        Get all available genes in the database.
        
        Returns:
            List of gene symbols
        """
        return self.database.get_all_gene_symbols()
    
    def get_available_phenotypes(self) -> List[str]:
        """
        Get all available phenotype IDs in the database.
        
        Returns:
            List of HPO IDs
        """
        return self.database.get_all_phenotype_ids()
    
    def update_gene_phenotype(self, gene: str, phenotype_id: str,
                            frequency: Optional[str] = None,
                            disease_id: Optional[str] = None) -> bool:
        """
        Add a new gene-phenotype association to the database.
        
        Args:
            gene: Gene symbol (e.g., 'AARS1')
            phenotype_id: HPO term ID (e.g., 'HP:0001939')
            frequency: Optional frequency information
            disease_id: Optional disease ID
            
        Returns:
            bool: True if the association was added, False if the phenotype doesn't exist
        """
        return self.database.update_gene_phenotype(gene, phenotype_id, 
                                                 frequency, disease_id)
    
    def save(self) -> None:
        """Save the gene-phenotype database to files."""
        saved_files = self.database.save()
        return saved_files 