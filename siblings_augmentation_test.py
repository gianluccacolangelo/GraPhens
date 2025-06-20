import logging
import os
from src.augmentation.hpo_augmentation import SiblingsAugmentationService, HPOAugmentationService
from src.core.types import Phenotype
from src.visualization.graphviz import GraphvizVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_multi_step_augmentation():
    """
    Tests a multi-step augmentation process:
    1. Augment with siblings.
    2. Augment the result with the full hierarchy (ancestors).
    Visualizes both steps.
    """
    # 1. Define initial phenotype(s)
    # "Atonic seizure" (HP:0001251) is a child of "Seizure" (HP:0001250).
    initial_phenotypes = [
        Phenotype(id="HP:0001251", name="Atonic seizure")
    ]
    logging.info(f"Initial phenotypes: {[p.name for p in initial_phenotypes]}")

    # --- Step 1: Siblings Augmentation ---
    logging.info("--- Running Siblings Augmentation ---")
    siblings_augmentation_service = SiblingsAugmentationService(data_dir="data/ontology")
    siblings_augmented_phenotypes = siblings_augmentation_service.augment(initial_phenotypes)
    logging.info(f"Phenotype count after siblings augmentation: {len(siblings_augmented_phenotypes)}")

    # Visualize the first step
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = GraphvizVisualizer(
        output_dir=output_dir,
        hpo_provider=siblings_augmentation_service.hpo_graph
    )

    output_path_step1 = visualizer.visualize_augmentation_result(
        augmented_phenotypes=siblings_augmented_phenotypes,
        initial_phenotypes=initial_phenotypes,
        title="1_Siblings_Augmentation_Result"
    )

    if output_path_step1:
        logging.info(f"Siblings augmentation visualization saved to: {output_path_step1}")
    else:
        logging.error("Failed to create siblings augmentation visualization.")

    # --- Step 2: HPO Hierarchy Augmentation ---
    logging.info("--- Running HPO Hierarchy (Ancestor) Augmentation ---")
    hpo_augmentation_service = HPOAugmentationService(data_dir="data/ontology", include_ancestors=True, include_descendants=False)
    
    # Feed the result of step 1 into step 2
    fully_augmented_phenotypes = hpo_augmentation_service.augment(siblings_augmented_phenotypes)
    logging.info(f"Phenotype count after full hierarchy augmentation: {len(fully_augmented_phenotypes)}")

    # Visualize the final result, still highlighting the original phenotypes
    output_path_step2 = visualizer.visualize_augmentation_result(
        augmented_phenotypes=fully_augmented_phenotypes,
        initial_phenotypes=initial_phenotypes,
        title="2_Full_Hierarchy_Augmentation_Result"
    )

    if output_path_step2:
        logging.info(f"Full hierarchy visualization saved to: {output_path_step2}")
    else:
        logging.error("Failed to create full hierarchy visualization.")

if __name__ == "__main__":
    test_multi_step_augmentation()
