#!/usr/bin/env python3
import os
import logging
from typing import List, Set, Optional

# Make sure 'src' is in the Python path.
# If running from project root, this should work.
# Consider adding project root to PYTHONPATH if issues arise.
try:
    from src.core.types import Phenotype
    from src.ontology.hpo_graph import HPOGraphProvider
    from src.visualization.graphviz import GraphvizVisualizer
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure you are running this script from the GraPhens project root,")
    print("or that the 'src' directory is in your PYTHONPATH.")
    exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
# Adjust HPO_DATA_DIR if your HPO data (e.g., hp.obo) is located elsewhere relative to project root.
HPO_DATA_DIR = "data/ontology"
VISUALIZATION_OUTPUT_DIR = "custom_phenotype_visualizations" # Output directory in project root
OUTPUT_FILE_TITLE = "Custom_Phenotype_Augmentation"
# Add a new title for the observed-only graph
OBSERVED_ONLY_GRAPH_TITLE = "Observed_Phenotypes_Only"

# HPO IDs provided by the user
USER_INITIAL_HPO_IDS_STR = "HP:0001260,HP:0002141,HP:0002167,HP:0001250,HP:0001025"

def get_phenotype_objects(hpo_ids: Set[str], provider: HPOGraphProvider) -> List[Phenotype]:
    """Creates a list of Phenotype objects from a set of HPO IDs."""
    phenotype_objects: List[Phenotype] = []
    for hpo_id in sorted(list(hpo_ids)): # Sort for consistent processing order
        term_data = provider.terms.get(hpo_id)
        if term_data:
            name = term_data.get("name", hpo_id)  # Default to ID if name not found
            phenotype_objects.append(Phenotype(id=hpo_id, name=name))
        else:
            logger.warning(f"Phenotype ID {hpo_id} not found in HPO provider's terms. Skipping object creation for this ID.")
    return phenotype_objects

def get_all_ancestors_and_self_ids(
    initial_hpo_ids: Set[str], 
    provider: HPOGraphProvider
) -> Set[str]:
    """
    Retrieves all unique HPO IDs consisting of the initial set and all their ancestors.
    This relies on provider.graph.successors(node_id) returning parent IDs,
    as observed in the existing GraphvizVisualizer code.
    """
    collected_ids: Set[str] = set()
    queue: List[str] = list(initial_hpo_ids)

    processed_for_queue: Set[str] = set()


    while queue:
        current_id = queue.pop(0)

        if current_id in processed_for_queue and current_id in collected_ids:
            continue
        
        processed_for_queue.add(current_id)

        # Ensure the term is known before trying to find parents or adding it
        if not provider.terms.get(current_id):
            if current_id in initial_hpo_ids: # Log only if it was an initial ID not found in terms
                 logger.warning(f"Initial HPO ID {current_id} is not in provider.terms. Cannot process.")
            else: # This means an ancestor ID from graph traversal isn't in terms, which is odd.
                 logger.warning(f"Ancestor HPO ID {current_id} (found via graph) is not in provider.terms. Skipping.")
            continue
        
        collected_ids.add(current_id)

        if not provider.graph.has_node(current_id):
            # Node is in terms but not in graph structure (e.g., obsolete, root, or data issue)
            # For HPO, roots like 'All' (HP:0000001) might not have 'successors' (parents)
            # logger.debug(f"HPO ID {current_id} is in terms but not in graph or has no parents. No further ancestors to trace via graph for this path.")
            continue
        
        try:
            # Assuming provider.graph.successors(node_id) returns parent IDs.
            # This is based on its usage in the project's GraphvizVisualizer.
            parent_ids = list(provider.graph.successors(current_id))
            for parent_id in parent_ids:
                if parent_id not in processed_for_queue: # Add to queue only if not processed
                    queue.append(parent_id)
        except Exception as e:
            logger.error(f"Error accessing parents for {current_id} from HPO graph: {e}. This might indicate an issue with the HPO graph data or structure.")
            
    return collected_ids

def main():
    """
    Main function to generate the phenotype visualization.
    """
    logger.info(f"Starting phenotype visualization script.")
    logger.info(f"Visualizing initial HPO IDs: {USER_INITIAL_HPO_IDS_STR}")

    initial_hpo_ids: Set[str] = {hpo_id.strip() for hpo_id in USER_INITIAL_HPO_IDS_STR.split(',') if hpo_id.strip()}

    if not initial_hpo_ids:
        logger.error("No valid HPO IDs provided. Exiting.")
        return

    # 1. Initialize HPOGraphProvider
    try:
        logger.info(f"Initializing HPOGraphProvider with data_dir='{HPO_DATA_DIR}'...")
        hpo_provider = HPOGraphProvider.get_instance(data_dir=HPO_DATA_DIR)
        hpo_provider.load()  # Crucial to load the data
        logger.info("HPOGraphProvider loaded successfully.")
        if not hpo_provider.terms or not hasattr(hpo_provider, 'graph') or not hpo_provider.graph:
            logger.error("HPOGraphProvider loaded, but 'terms' or 'graph' data is missing/empty. Please check HPO data files (e.g., hp.obo) in '{HPO_DATA_DIR}'.")
            return
    except FileNotFoundError:
        logger.error(f"HPO data directory or essential files not found at '{HPO_DATA_DIR}'. Please ensure correct path and data.")
        return
    except Exception as e:
        logger.error(f"Failed to initialize or load HPOGraphProvider: {e}", exc_info=True)
        return

    # 2. Create Phenotype objects for the initial set
    # These are the phenotypes to be highlighted
    initial_phenotypes: List[Phenotype] = get_phenotype_objects(initial_hpo_ids, hpo_provider)
    if not initial_phenotypes:
        logger.error("No valid Phenotype objects could be created for the initial set. This might be due to all provided IDs not being found in the HPO data. Exiting.")
        return
    logger.info(f"Created {len(initial_phenotypes)} initial Phenotype objects: {[p.id for p in initial_phenotypes]}")
    
    # Filter initial_hpo_ids to only those for which Phenotype objects were successfully created
    valid_initial_hpo_ids = {p.id for p in initial_phenotypes}

    # 3. Determine the full set of phenotypes for the visualization (initial + all their ancestors)
    logger.info("Determining augmented set (initial phenotypes + all their ancestors)...")
    all_related_ids: Set[str] = get_all_ancestors_and_self_ids(valid_initial_hpo_ids, hpo_provider)
    
    if not all_related_ids:
        logger.error("Could not determine any related HPO IDs (initial + ancestors). Exiting.")
        return
    logger.info(f"Found {len(all_related_ids)} unique HPO IDs for the augmented set.")

    augmented_phenotypes: List[Phenotype] = get_phenotype_objects(all_related_ids, hpo_provider)
    if not augmented_phenotypes:
        logger.error("No Phenotype objects could be created for the augmented set. Exiting.")
        return
    logger.info(f"Created {len(augmented_phenotypes)} augmented Phenotype objects for visualization.")

    # 4. Initialize GraphvizVisualizer
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    visualizer = GraphvizVisualizer(output_dir=VISUALIZATION_OUTPUT_DIR, hpo_provider=hpo_provider)
    logger.info(f"GraphvizVisualizer initialized. Output will be in '{os.path.abspath(VISUALIZATION_OUTPUT_DIR)}'.")

    # 5. Visualize the result
    # Graph 1: Observed Phenotypes Only
    logger.info(f"Generating Graph 1: Observed Phenotypes Only, titled: '{OBSERVED_ONLY_GRAPH_TITLE}'...")
    try:
        # For this graph, we only want to see the initial_phenotypes and their relationships.
        # If initial_phenotypes themselves have parent/child relationships *within that set*, they'll be shown.
        # We pass initial_phenotypes as both 'phenotypes' and 'initial_phenotypes'
        # so they are all rendered with the 'initial' style.
        image_path_observed_only = visualizer.visualize_hierarchy(
            phenotypes=initial_phenotypes, # Only show these nodes
            initial_phenotypes=initial_phenotypes, # Highlight all of them (or use the 'initial' color)
            title=OBSERVED_ONLY_GRAPH_TITLE,
            hpo_provider=hpo_provider # Though already set in visualizer, explicit for clarity
        )
        
        if image_path_observed_only:
            abs_image_path_observed_only = os.path.abspath(image_path_observed_only)
            logger.info(f"SUCCESS: Graph 1 (Observed Only) saved to: {abs_image_path_observed_only}")
            print(f"Graph 1 (Observed Only) generated successfully!")
            print(f"Output image: {abs_image_path_observed_only}")
        else:
            logger.error(f"Graph 1 (Observed Only) generation did not return a path. Check logs from GraphvizVisualizer.")
            print("Graph 1 (Observed Only) generation may have failed. Please check the logs.")

    except ImportError:
        logger.error(f"The 'graphviz' Python package is not installed for Graph 1. Please install it and Graphviz system binaries.")
        print("ERROR for Graph 1: 'graphviz' Python package or system utility is missing.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Graph 1 visualization: {e}", exc_info=True)
        print(f"An unexpected error occurred during Graph 1 visualization. Check logs for details: {e}")

    print("\n" + "-"*30 + "\n") # Separator in console output

    # Graph 2: Augmented Phenotype Hierarchy (initial + ancestors)
    logger.info(f"Generating Graph 2: Augmented Phenotype Hierarchy, titled: '{OUTPUT_FILE_TITLE}'...")
    try:
        image_path_augmented = visualizer.visualize_augmentation_result(
            augmented_phenotypes=augmented_phenotypes,
            initial_phenotypes=initial_phenotypes,
            title=OUTPUT_FILE_TITLE
            # hpo_provider is already set in the visualizer instance
        )
        
        if image_path_augmented:
            abs_image_path_augmented = os.path.abspath(image_path_augmented)
            logger.info(f"SUCCESS: Graph 2 (Augmented) saved to: {abs_image_path_augmented}")
            print(f"Graph 2 (Augmented Hierarchy) generated successfully!")
            print(f"Output image: {abs_image_path_augmented}")
        else:
            # This case might occur if graphviz itself fails internally but doesn't raise Python error to here.
            # The GraphvizVisualizer logs errors if graphviz package is missing.
            logger.error("Graph 2 (Augmented) generation did not return a path. Check logs from GraphvizVisualizer for specific errors (e.g., Graphviz system library not found or other Graphviz error).")
            print("Graph 2 (Augmented Hierarchy) generation may have failed. Please check the logs.")

    except ImportError: # Specifically for 'import graphviz' failing inside the visualizer
        logger.error("The 'graphviz' Python package is not installed for Graph 2. Please install it (e.g., 'pip install graphviz') and ensure the Graphviz system binaries are also installed.")

if __name__ == "__main__":
    main() 