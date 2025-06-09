#!/usr/bin/env python3
"""
Integration tests for the Gene-Phenotype Database module.

These tests demonstrate how the gene-phenotype database integrates with other
GraPhens modules to create a complete workflow.
"""

import os
import sys
import unittest
import tempfile
import shutil
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.gene_phenotype_db import HPOAGenePhenotypeDatabase
from src.core.types import Phenotype
from src.database.gene_phenotype_analysis import analyze_gene_phenotypes

class TestDatabaseIntegration(unittest.TestCase):
    """Test integration of the database module with other components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        cls.database_path = os.path.join(cls.temp_dir, "test_phenotype.hpoa")
        
        # Create a simple test database file
        test_data = """ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id
10\tNAT2\tHP:0000007\tAutosomal recessive inheritance\t-\tOMIM:243400
10\tNAT2\tHP:0001939\tAbnormality of metabolism/homeostasis\t-\tOMIM:243400
16\tAARS1\tHP:0002460\tDistal muscle weakness\t15/15\tOMIM:613287
16\tAARS1\tHP:0002451\tLimb dystonia\t3/3\tOMIM:616339
16\tAARS1\tHP:0008619\tBilateral sensorineural hearing impairment\t-\tOMIM:613287
"""
        with open(cls.database_path, 'w') as f:
            f.write(test_data)
            
        # Initialize the database
        cls.db = HPOAGenePhenotypeDatabase(
            genes_to_phenotypes_path=cls.database_path
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory and its contents
        shutil.rmtree(cls.temp_dir)
    
    def test_get_phenotypes_for_gene(self):
        """Test retrieving phenotypes for a gene."""
        # Get phenotypes for AARS1
        phenotypes = self.db.get_phenotypes_for_gene("AARS1")
        
        # Verify results
        self.assertEqual(len(phenotypes), 3, "Should find 3 phenotypes for AARS1")
        
        # Check that all phenotypes have the correct format
        for p in phenotypes:
            self.assertIsInstance(p, Phenotype, "Result should be Phenotype objects")
            self.assertTrue(p.id.startswith("HP:"), "Phenotype ID should start with HP:")
            self.assertIn("gene_symbol", p.metadata, "Metadata should include gene_symbol")
            self.assertEqual(p.metadata["gene_symbol"], "AARS1", 
                             "Gene symbol in metadata should match query")
    
    def test_get_genes_for_phenotype(self):
        """Test retrieving genes for a phenotype."""
        # Get genes for HP:0000007 (Autosomal recessive inheritance)
        genes = self.db.get_genes_for_phenotype("HP:0000007")
        
        # Verify results
        self.assertIn("NAT2", genes, "NAT2 should be associated with HP:0000007")
        
    def test_get_frequency(self):
        """Test retrieving frequency information."""
        # Get frequency for AARS1 - HP:0002460 - OMIM:613287
        frequency = self.db.get_frequency("AARS1", "HP:0002460", "OMIM:613287")
        
        # Verify results
        self.assertEqual(frequency, "15/15", "Frequency should be 15/15")
    
    def test_get_diseases_for_gene(self):
        """Test retrieving diseases for a gene."""
        # Get diseases for AARS1
        diseases = self.db.get_diseases_for_gene("AARS1")
        
        # Verify results
        self.assertIn("OMIM:613287", diseases, "OMIM:613287 should be associated with AARS1")
        self.assertIn("OMIM:616339", diseases, "OMIM:616339 should be associated with AARS1")
    
    def test_get_genes_for_disease(self):
        """Test retrieving genes for a disease."""
        # Get genes for OMIM:243400
        genes = self.db.get_genes_for_disease("OMIM:243400")
        
        # Verify results
        self.assertIn("NAT2", genes, "NAT2 should be associated with OMIM:243400")

class TestDatabaseWithOtherComponents(unittest.TestCase):
    """
    Test integration of the database module with other components.
    
    Note: These tests are marked as skip by default since they require 
    the HPO ontology files to be present. To run them, download the HPO files
    to data/ontology and remove the skip decorator.
    """
    
    @unittest.skip("Requires HPO ontology files")
    def test_analyze_gene_phenotypes(self):
        """Test the gene phenotype analysis workflow."""
        # Set up test environment
        test_dir = tempfile.mkdtemp()
        database_path = os.path.join(test_dir, "test_phenotype.hpoa")
        
        # Create a simple test database file
        test_data = """ncbi_gene_id\tgene_symbol\thpo_id\thpo_name\tfrequency\tdisease_id
10\tNAT2\tHP:0000007\tAutosomal recessive inheritance\t-\tOMIM:243400
10\tNAT2\tHP:0001939\tAbnormality of metabolism/homeostasis\t-\tOMIM:243400
16\tAARS1\tHP:0002460\tDistal muscle weakness\t15/15\tOMIM:613287
16\tAARS1\tHP:0002451\tLimb dystonia\t3/3\tOMIM:616339
16\tAARS1\tHP:0008619\tBilateral sensorineural hearing impairment\t-\tOMIM:613287
"""
        with open(database_path, 'w') as f:
            f.write(test_data)
        
        try:
            # Run the analysis
            results = analyze_gene_phenotypes(
                gene_symbol="AARS1",
                database_path=database_path,
                ontology_dir="data/ontology",  # Requires actual HPO files
                augment=True,
                verbose=False
            )
            
            # Verify results
            self.assertTrue(results["success"], "Analysis should succeed")
            self.assertEqual(results["gene_symbol"], "AARS1", "Gene symbol should match")
            self.assertEqual(results["phenotype_count"]["original"], 3, 
                             "Should find 3 original phenotypes")
            self.assertGreater(results["phenotype_count"]["augmented"], 3, 
                               "Augmented count should be greater than original")
            
            # Check graph properties
            self.assertIn("graph", results, "Results should include graph details")
            self.assertGreater(results["graph"]["node_count"], 0, "Graph should have nodes")
            self.assertGreater(results["graph"]["edge_count"], 0, "Graph should have edges")
            self.assertEqual(results["graph"]["feature_dim"], 768, 
                             "Feature dimension should match TFIDFEmbeddingStrategy")
            
        finally:
            # Clean up
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main() 