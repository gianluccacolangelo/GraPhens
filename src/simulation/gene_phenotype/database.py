from typing import List, Dict, Optional, Set, Tuple
import os
import pandas as pd
from src.core.types import Phenotype


class GenePhenotypeDatabase:
    """
    A class for accessing gene-phenotype relationships from HPO database files.
    
    This database provides methods to query relationships between genes and phenotypes,
    including frequency information when available.
    """
    
    def __init__(self, data_dir: str = "data/gene_phenotype"):
        """
        Initialize the gene-phenotype database.
        
        Args:
            data_dir: Directory containing the gene-phenotype database files
        """
        self.data_dir = data_dir
        self._genes_to_phenotype = None
        self._phenotype_to_genes = None
        
    def load(self) -> None:
        """Load the gene-phenotype database files."""
        self._load_genes_to_phenotype()
        self._load_phenotype_to_genes()
        
    def _load_genes_to_phenotype(self) -> None:
        """Load the genes_to_phenotype.txt file."""
        file_path = os.path.join(self.data_dir, "genes_to_phenotype.txt")
        self._genes_to_phenotype = pd.read_csv(
            file_path, 
            sep='\t',
            dtype={
                'ncbi_gene_id': str,
                'gene_symbol': str,
                'hpo_id': str,
                'hpo_name': str,
                'frequency': str,
                'disease_id': str
            }
        )
    
    def _load_phenotype_to_genes(self) -> None:
        """Load the phenotype_to_genes.txt file."""
        file_path = os.path.join(self.data_dir, "phenotype_to_genes.txt")
        self._phenotype_to_genes = pd.read_csv(
            file_path, 
            sep='\t',
            dtype={
                'hpo_id': str,
                'hpo_name': str,
                'ncbi_gene_id': str,
                'gene_symbol': str,
                'disease_id': str
            }
        )
    
    def get_phenotypes_for_gene(self, gene_symbol: str) -> List[Phenotype]:
        """
        Get all phenotypes associated with a gene.
        
        Args:
            gene_symbol: The gene symbol (e.g., 'AARS1')
            
        Returns:
            List of Phenotype objects associated with the gene
        """
        if self._genes_to_phenotype is None:
            self.load()
            
        filtered = self._genes_to_phenotype[
            self._genes_to_phenotype['gene_symbol'] == gene_symbol
        ]
        
        phenotypes = []
        for _, row in filtered.iterrows():
            metadata = {
                'gene_symbol': row['gene_symbol'],
                'ncbi_gene_id': row['ncbi_gene_id'],
                'frequency': row['frequency'],
                'disease_id': row['disease_id']
            }
            
            phenotype = Phenotype(
                id=row['hpo_id'],
                name=row['hpo_name'],
                metadata=metadata
            )
            phenotypes.append(phenotype)
            
        return phenotypes
    
    def get_genes_for_phenotype(self, hpo_id: str, include_ancestors: bool = False) -> List[Dict[str, str]]:
        """
        Get all genes associated with a phenotype.
        
        Args:
            hpo_id: The HPO term ID (e.g., 'HP:0001939')
            include_ancestors: Whether to include genes associated with ancestor terms
                              (use phenotype_to_genes.txt when True)
            
        Returns:
            List of gene info dictionaries with keys:
            - gene_symbol
            - ncbi_gene_id
            - disease_id
        """
        if self._genes_to_phenotype is None or self._phenotype_to_genes is None:
            self.load()
            
        if include_ancestors:
            df = self._phenotype_to_genes
        else:
            df = self._genes_to_phenotype
            
        filtered = df[df['hpo_id'] == hpo_id]
        
        genes = []
        for _, row in filtered.iterrows():
            gene_info = {
                'gene_symbol': row['gene_symbol'],
                'ncbi_gene_id': row['ncbi_gene_id'],
                'disease_id': row['disease_id']
            }
            genes.append(gene_info)
            
        return genes
    
    def get_frequency_data(self, gene_symbol: str, hpo_id: Optional[str] = None, 
                          disease_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get frequency information for phenotypes associated with a gene/disease.
        
        Args:
            gene_symbol: The gene symbol (e.g., 'AARS1')
            hpo_id: Optional HPO term ID to filter by specific phenotype
            disease_id: Optional disease ID to filter by specific disease
            
        Returns:
            List of dictionaries with keys:
            - hpo_id
            - hpo_name
            - frequency
            - disease_id
        """
        if self._genes_to_phenotype is None:
            self.load()
            
        filtered = self._genes_to_phenotype[
            self._genes_to_phenotype['gene_symbol'] == gene_symbol
        ]
        
        if hpo_id:
            filtered = filtered[filtered['hpo_id'] == hpo_id]
            
        if disease_id:
            filtered = filtered[filtered['disease_id'] == disease_id]
            
        results = []
        for _, row in filtered.iterrows():
            frequency_data = {
                'hpo_id': row['hpo_id'],
                'hpo_name': row['hpo_name'],
                'frequency': row['frequency'],
                'disease_id': row['disease_id']
            }
            results.append(frequency_data)
            
        return results
    
    def get_all_gene_symbols(self) -> List[str]:
        """Get all unique gene symbols in the database."""
        if self._genes_to_phenotype is None:
            self.load()
        
        return sorted(self._genes_to_phenotype['gene_symbol'].unique().tolist())
    
    def get_all_phenotype_ids(self) -> List[str]:
        """Get all unique HPO IDs in the database."""
        if self._genes_to_phenotype is None:
            self.load()
        
        return sorted(self._genes_to_phenotype['hpo_id'].unique().tolist())
    
    def get_diseases_for_gene(self, gene_symbol: str) -> List[str]:
        """Get all diseases associated with a gene."""
        if self._genes_to_phenotype is None:
            self.load()
            
        filtered = self._genes_to_phenotype[
            self._genes_to_phenotype['gene_symbol'] == gene_symbol
        ]
        
        return sorted(filtered['disease_id'].unique().tolist())
    
    def update_gene_phenotype(self, gene_symbol: str, hpo_id: str, 
                            frequency: Optional[str] = None,
                            disease_id: Optional[str] = None) -> bool:
        """
        Add a new gene-phenotype association to the database.
        
        Args:
            gene_symbol: The gene symbol (e.g., 'AARS1')
            hpo_id: The HPO term ID (e.g., 'HP:0001939')
            frequency: Optional frequency information
            disease_id: Optional disease ID
            
        Returns:
            bool: True if the association was added, False if the phenotype doesn't exist
        """
        if self._genes_to_phenotype is None:
            self.load()
            
        # Check if the phenotype exists in the database
        if hpo_id not in self._genes_to_phenotype['hpo_id'].values:
            return False
            
        # Get the HPO name from existing records
        hpo_name = self._genes_to_phenotype[
            self._genes_to_phenotype['hpo_id'] == hpo_id
        ]['hpo_name'].iloc[0]
        
        # Get NCBI gene ID if it exists, otherwise use gene symbol as ID
        existing_gene = self._genes_to_phenotype[
            self._genes_to_phenotype['gene_symbol'] == gene_symbol
        ]
        ncbi_gene_id = (existing_gene['ncbi_gene_id'].iloc[0] 
                        if not existing_gene.empty else gene_symbol)
        
        # Create new row
        new_row = pd.DataFrame({
            'ncbi_gene_id': [ncbi_gene_id],
            'gene_symbol': [gene_symbol],
            'hpo_id': [hpo_id],
            'hpo_name': [hpo_name],
            'frequency': [frequency if frequency else ''],
            'disease_id': [disease_id if disease_id else '']
        })
        
        # Append to both dataframes
        self._genes_to_phenotype = pd.concat(
            [self._genes_to_phenotype, new_row], ignore_index=True
        )
        self._phenotype_to_genes = pd.concat(
            [self._phenotype_to_genes, new_row[['hpo_id', 'hpo_name', 'ncbi_gene_id', 
                                               'gene_symbol', 'disease_id']]], 
            ignore_index=True
        )
        
        return True 

    def save(self) -> None:
        """Save the gene-phenotype database to files."""
        saved_files = []
        
        if self._genes_to_phenotype is not None:
            file_path = os.path.join(self.data_dir, "genes_to_phenotype.txt")
            self._genes_to_phenotype.to_csv(file_path, sep='\t', index=False)
            saved_files.append(file_path)
        
        if self._phenotype_to_genes is not None:
            file_path = os.path.join(self.data_dir, "phenotype_to_genes.txt")
            self._phenotype_to_genes.to_csv(file_path, sep='\t', index=False)
            saved_files.append(file_path)
            
        return saved_files 