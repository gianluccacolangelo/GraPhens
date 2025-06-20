import logging
import os
from src.augmentation.hpo_augmentation import NHopAugmentationService, HPOAugmentationService
from src.core.types import Phenotype
from src.visualization.graphviz import GraphvizVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_nhop_and_full_augmentation():
    """
    Tests a multi-step augmentation process:
    1. Augment with N-hop neighbors.
    2. Augment the result with the full hierarchy (ancestors).
    Visualizes both steps.
    """
    # 1. Define initial phenotype(s)
    initial_phenotypes = [
        Phenotype(id="HP:0001251", name="Atonic seizure")
    ]
    logging.info(f"Initial phenotypes: {[p.name for p in initial_phenotypes]}")
    
    # Define N-hops for the test
    n_hops = 2
    logging.info(f"Using n_hops = {n_hops}")


    # --- Step 1: N-Hop Augmentation ---
    logging.info(f"--- Running {n_hops}-Hop Augmentation ---")
    nhop_augmentation_service = NHopAugmentationService(n_hops=n_hops, data_dir="data/ontology")
    nhop_augmented_phenotypes = nhop_augmentation_service.augment(initial_phenotypes)
    logging.info(f"Phenotype count after {n_hops}-hop augmentation: {len(nhop_augmented_phenotypes)}")

    # Visualize the first step
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = GraphvizVisualizer(
        output_dir=output_dir,
        hpo_provider=nhop_augmentation_service.hpo_graph
    )

    output_path_step1 = visualizer.visualize_augmentation_result(
        augmented_phenotypes=nhop_augmented_phenotypes,
        initial_phenotypes=initial_phenotypes,
        title=f"1_{n_hops}-Hop_Augmentation_Result"
    )

    if output_path_step1:
        logging.info(f"{n_hops}-hop augmentation visualization saved to: {output_path_step1}")
    else:
        logging.error(f"Failed to create {n_hops}-hop augmentation visualization.")

    # --- Step 2: HPO Hierarchy Augmentation ---
    logging.info("--- Running HPO Hierarchy (Ancestor) Augmentation ---")
    hpo_augmentation_service = HPOAugmentationService(data_dir="data/ontology", include_ancestors=True, include_descendants=False)
    
    # Feed the result of step 1 into step 2
    fully_augmented_phenotypes = hpo_augmentation_service.augment(nhop_augmented_phenotypes)
    logging.info(f"Phenotype count after full hierarchy augmentation: {len(fully_augmented_phenotypes)}")

    # Visualize the final result, still highlighting the original phenotypes
    output_path_step2 = visualizer.visualize_augmentation_result(
        augmented_phenotypes=fully_augmented_phenotypes,
        initial_phenotypes=initial_phenotypes,
        title=f"2_Full_Hierarchy_from_{n_hops}-Hop_Result"
    )

    if output_path_step2:
        logging.info(f"Full hierarchy visualization saved to: {output_path_step2}")
    else:
        logging.error("Failed to create full hierarchy visualization.")

if __name__ == "__main__":
    test_nhop_and_full_augmentation() 