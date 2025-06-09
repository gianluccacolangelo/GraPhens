#!/usr/bin/env python
"""
Demonstration script for the IndexAlignmentChecker.

This script shows:
1. How the validation works when components are properly aligned
2. How it detects misalignments in various scenarios
3. How to use the validation with the graph assembly process
"""

import numpy as np
import argparse
from src.core.types import Phenotype
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import TFIDFEmbeddingStrategy
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.graph.assembler import StandardGraphAssembler
from src.graph.validation import IndexAlignmentChecker, ValidationError
from src.ontology.hpo_graph import HPOGraphProvider
from typing import List, Tuple

def create_test_phenotypes() -> List[Phenotype]:
    """Create a test set of phenotypes."""
    return [
        Phenotype(id="HP:0001250", name="Seizure"),
        Phenotype(id="HP:0001251", name="Atonic seizure"),
        Phenotype(id="HP:0001257", name="Absence seizure"),
        Phenotype(id="HP:0000118", name="Phenotypic abnormality"),
        Phenotype(id="HP:0000707", name="Abnormality of the nervous system")
    ]

def demonstrate_correct_alignment():
    """Demonstrate validation when components are correctly aligned."""
    print("\n[1] Demonstrating correct alignment validation:")
    
    # Create phenotypes
    phenotypes = create_test_phenotypes()
    print(f"Created {len(phenotypes)} test phenotypes")
    
    # Create node features (correctly aligned)
    strategy = TFIDFEmbeddingStrategy(max_features=10)
    context = EmbeddingContext(strategy)
    node_features = context.embed_phenotypes(phenotypes)
    print(f"Generated node features with shape {node_features.shape}")
    
    # Create edge index (correctly aligned)
    hpo_provider = HPOGraphProvider()
    hpo_provider.load()
    builder = HPOAdjacencyListBuilder(hpo_provider)
    edge_index = builder.build(phenotypes)
    print(f"Generated edge index with shape {edge_index.shape}")
    
    # Validate components
    try:
        is_valid = IndexAlignmentChecker.check_components(
            phenotypes, node_features, edge_index
        )
        print(f"✅ Validation successful: {is_valid}")
    except ValidationError as e:
        print(f"❌ Validation failed: {str(e)}")
    
    # Assemble graph with validation
    assembler = StandardGraphAssembler(validate=True)
    try:
        graph = assembler.assemble(
            phenotypes=phenotypes,
            node_features=node_features,
            edge_index=edge_index
        )
        print(f"✅ Graph assembly successful")
    except ValidationError as e:
        print(f"❌ Graph assembly failed: {str(e)}")

def demonstrate_misaligned_features():
    """Demonstrate validation when node features are misaligned."""
    print("\n[2] Demonstrating detection of misaligned node features:")
    
    # Create phenotypes
    phenotypes = create_test_phenotypes()
    
    # Create node features with wrong number of rows
    node_features = np.random.random((len(phenotypes) - 1, 10))
    print(f"Created misaligned node features with shape {node_features.shape}")
    
    # Create valid edge index
    edge_index = np.array([[0, 1], [1, 2]]).T
    
    # Validate components
    try:
        IndexAlignmentChecker.check_components(
            phenotypes, node_features, edge_index
        )
        print(f"❌ Validation incorrectly passed")
    except ValidationError as e:
        print(f"✅ Detected misalignment: {str(e)}")

def demonstrate_misaligned_edges():
    """Demonstrate validation when edge indices are out of bounds."""
    print("\n[3] Demonstrating detection of out-of-bounds edge indices:")
    
    # Create phenotypes
    phenotypes = create_test_phenotypes()
    
    # Create valid node features
    node_features = np.random.random((len(phenotypes), 10))
    
    # Create edge index with out-of-bounds index
    edge_index = np.array([[0, 1, len(phenotypes)], [1, 2, 0]]).T
    print(f"Created edge index with out-of-bounds value: {edge_index}")
    
    # Validate components
    try:
        IndexAlignmentChecker.check_components(
            phenotypes, node_features, edge_index
        )
        print(f"❌ Validation incorrectly passed")
    except ValidationError as e:
        print(f"✅ Detected out-of-bounds: {str(e)}")

def demonstrate_user_friendly_validation():
    """Demonstrate the user-friendly validation interface."""
    print("\n[4] Demonstrating user-friendly validation interface:")
    
    # Create phenotypes and valid components
    phenotypes = create_test_phenotypes()
    node_features = np.random.random((len(phenotypes), 10))
    edge_index = np.array([[0, 1], [1, 2]]).T
    
    # Test valid case
    is_valid, message = IndexAlignmentChecker.validate_embedding_adjacency_alignment(
        phenotypes, node_features, edge_index
    )
    print(f"Valid case: {is_valid}, Message: {message}")
    
    # Test invalid case (wrong feature count)
    is_valid, message = IndexAlignmentChecker.validate_embedding_adjacency_alignment(
        phenotypes, np.random.random((len(phenotypes) - 1, 10)), edge_index
    )
    print(f"Invalid case (features): {is_valid}, Message: {message}")
    
    # Test invalid case (out of bounds edge)
    is_valid, message = IndexAlignmentChecker.validate_embedding_adjacency_alignment(
        phenotypes, node_features, np.array([[0, 10], [1, 0]]).T
    )
    print(f"Invalid case (edges): {is_valid}, Message: {message}")

def demonstrate_modified_phenotype_order():
    """Demonstrate alignment validation with modified phenotype order."""
    print("\n[5] Demonstrating impact of phenotype order:")
    
    # Create original phenotypes
    original_phenotypes = create_test_phenotypes()
    
    # Create node features for original phenotypes
    strategy = TFIDFEmbeddingStrategy(max_features=10)
    context = EmbeddingContext(strategy)
    node_features = context.embed_phenotypes(original_phenotypes)
    
    # Create edge index for original phenotypes
    hpo_provider = HPOGraphProvider()
    hpo_provider.load()
    builder = HPOAdjacencyListBuilder(hpo_provider)
    edge_index = builder.build(original_phenotypes)
    
    # Now shuffle the phenotypes
    import random
    shuffled_phenotypes = original_phenotypes.copy()
    random.shuffle(shuffled_phenotypes)
    
    print("Original phenotype order:")
    for i, p in enumerate(original_phenotypes):
        print(f"  {i}: {p.id} - {p.name}")
    
    print("Shuffled phenotype order:")
    for i, p in enumerate(shuffled_phenotypes):
        print(f"  {i}: {p.id} - {p.name}")
    
    # Try to validate with shuffled phenotypes using the new method
    try:
        IndexAlignmentChecker.check_consistent_phenotype_order(
            original_phenotypes, shuffled_phenotypes
        )
        print(f"❌ Using phenotype order validation didn't detect the shuffle (it should have)")
    except ValidationError as e:
        print(f"✅ Detected misalignment from changed phenotype order: {str(e)}")
    
    # Try to validate with the regular check_components method
    try:
        IndexAlignmentChecker.check_components(
            shuffled_phenotypes, node_features, edge_index, original_phenotypes
        )
        print(f"❌ Using different phenotype order still didn't fail validation with basic check")
    except ValidationError as e:
        print(f"✅ Basic check now detects misalignment: {str(e)}")
    
    # Create new edge index with shuffled phenotypes
    new_edge_index = builder.build(shuffled_phenotypes)
    
    # Create new node features with shuffled phenotypes
    new_node_features = context.embed_phenotypes(shuffled_phenotypes)
    
    # This should validate correctly
    try:
        IndexAlignmentChecker.check_components(
            shuffled_phenotypes, new_node_features, new_edge_index
        )
        print(f"✅ Validation passes with consistent phenotype order")
    except ValidationError as e:
        print(f"❌ Validation failed with consistent phenotype order: {str(e)}")

def demonstrate_with_augmentation():
    """Demonstrate validation with augmented phenotypes including ancestors."""
    print("\n[6] Demonstrating validation with HPO augmentation:")
    
    # Create initial phenotypes
    initial_phenotypes = create_test_phenotypes()
    print(f"Created {len(initial_phenotypes)} initial phenotypes")
    
    # Augment with ancestors
    from src.augmentation.hpo_augmentation import HPOAugmentationService
    augmentation_service = HPOAugmentationService(
        include_ancestors=True,
        include_descendants=False
    )
    
    augmented_phenotypes = augmentation_service.augment(initial_phenotypes)
    print(f"Augmented to {len(augmented_phenotypes)} phenotypes with ancestors")
    
    # Create node features for the augmented set
    strategy = TFIDFEmbeddingStrategy(max_features=10)
    context = EmbeddingContext(strategy)
    node_features = context.embed_phenotypes(augmented_phenotypes)
    print(f"Generated node features with shape {node_features.shape}")
    
    # Create edge index for the augmented set
    hpo_provider = HPOGraphProvider()
    hpo_provider.load()
    builder = HPOAdjacencyListBuilder(hpo_provider)
    edge_index = builder.build(augmented_phenotypes)
    print(f"Generated edge index with shape {edge_index.shape}")
    
    # Validate the components
    try:
        IndexAlignmentChecker.check_components(
            augmented_phenotypes, node_features, edge_index
        )
        print(f"✅ Validation successful with augmented phenotypes")
    except ValidationError as e:
        print(f"❌ Validation failed: {str(e)}")
    
    # Now let's test with a shuffled subset
    import random
    subset_size = min(10, len(augmented_phenotypes))
    original_subset = augmented_phenotypes[:subset_size]
    shuffled_subset = original_subset.copy()
    random.shuffle(shuffled_subset)
    
    print(f"Created subset of {len(original_subset)} phenotypes and shuffled it")
    
    # Generate features and edges for the original subset
    original_subset_features = context.embed_phenotypes(original_subset)
    original_subset_edges = builder.build(original_subset)
    
    # Generate features and edges for the shuffled subset
    shuffled_subset_features = context.embed_phenotypes(shuffled_subset)
    shuffled_subset_edges = builder.build(shuffled_subset)
    
    # This should validate correctly
    try:
        IndexAlignmentChecker.check_components(
            shuffled_subset, shuffled_subset_features, shuffled_subset_edges
        )
        print(f"✅ Validation passes with consistent shuffled subset")
    except ValidationError as e:
        print(f"❌ Validation failed with consistent shuffled subset: {str(e)}")
    
    # Now test with mixed components (shuffled phenotypes but original edges)
    # This SHOULD fail
    try:
        # Using shuffled phenotypes with original subset edges
        IndexAlignmentChecker.check_consistent_phenotype_order(
            original_subset, shuffled_subset
        )
        print(f"❌ Phenotype order validation didn't detect mismatched components (it should have)")
    except ValidationError as e:
        print(f"✅ Correctly detected misalignment with mixed components: {str(e)}")
    
    try:
        # Using basic check_components validation
        IndexAlignmentChecker.check_components(
            shuffled_subset, shuffled_subset_features, original_subset_edges, original_subset
        )
        print(f"❌ Basic validation didn't detect mismatched components (it should have)")
    except ValidationError as e:
        print(f"✅ Basic validation now detects misalignment: {str(e)}")

def main():
    """Run the demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate index alignment validation")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6], 
                        help="Run a specific test (1-6)")
    args = parser.parse_args()
    
    print("===== INDEX ALIGNMENT VALIDATION DEMONSTRATION =====")
    
    if args.test == 1 or args.test is None:
        demonstrate_correct_alignment()
    
    if args.test == 2 or args.test is None:
        demonstrate_misaligned_features()
    
    if args.test == 3 or args.test is None:
        demonstrate_misaligned_edges()
    
    if args.test == 4 or args.test is None:
        demonstrate_user_friendly_validation()
    
    if args.test == 5 or args.test is None:
        demonstrate_modified_phenotype_order()
    
    if args.test == 6 or args.test is None:
        demonstrate_with_augmentation()
    
    print("\n===== DEMONSTRATION COMPLETE =====")

if __name__ == "__main__":
    main() 