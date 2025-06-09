import logging
import pandas as pd
import torch
import json
from pathlib import Path
import sys

# Ensure src directory is in path if script is run from validation_arena directly,
# though it's recommended to run from project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.graphens import GraPhens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_finetuning_set():
    """
    Generates a PyTorch Geometric fine-tuning dataset from the DDD.dataset.tsv file.
    """
    script_dir = Path(__file__).resolve().parent
    input_tsv_path = script_dir / "DDD.dataset.tsv"
    output_pt_path = script_dir / "DDD_finetuning_dataset.pt"
    output_label_map_path = script_dir / "DDD_gene_to_label.json"

    logger.info(f"Starting dataset generation from: {input_tsv_path}")

    # --- 1. Load Data from TSV ---
    if not input_tsv_path.exists():
        logger.error(f"Input file not found: {input_tsv_path}")
        return

    try:
        df = pd.read_csv(input_tsv_path, sep='\t')
        logger.info(f"Successfully read {len(df)} rows from {input_tsv_path}.")
    except Exception as e:
        logger.error(f"Error reading TSV file: {e}")
        return

    # --- 2. Prepare Data for GraPhens ---
    patient_phenotypes_map = {}
    sample_to_gene_map = {}

    for idx, row in df.iterrows():
        sample_id = str(row.get('ID', f"sample_{idx}")) # Use ID or generate one
        gene_symbol = str(row.get('Gene'))
        phenotype_str = str(row.get('Phenotype', ''))

        if not gene_symbol or gene_symbol.lower() == 'nan':
            logger.warning(f"Skipping row {idx+2} due to missing gene symbol. Sample ID: {sample_id}")
            continue

        phenotype_list = [hp.strip() for hp in phenotype_str.split(',') if hp.strip() and hp.strip().startswith("HP:")]
        
        if not phenotype_list:
            logger.warning(f"Sample {sample_id} (Gene: {gene_symbol}) has no valid HP phenotypes after parsing. Skipping.")
            continue
            
        patient_phenotypes_map[sample_id] = phenotype_list
        sample_to_gene_map[sample_id] = gene_symbol

    if not patient_phenotypes_map:
        logger.error("No valid samples found after parsing the TSV file. Exiting.")
        return
    logger.info(f"Prepared {len(patient_phenotypes_map)} samples with valid phenotypes for GraPhens.")

    # --- 3. Instantiate GraPhens ---
    try:
        # Assumes GraPhens default data_dir="data/ontology" and embeddings from "data/embeddings"
        # are accessible from the current working directory (project root).
        g = GraPhens()
        logger.info(f"GraPhens initialized. Ontology data dir: {g.data_dir}")
        embedding_config = g.config.get('embedding', {})
        logger.info(f"GraPhens embedding type: {embedding_config.get('type')}")
        if embedding_config.get('type') == 'lookup':
            # Check if embedding_dict is loaded and has keys
            # Due to complexity of GraPhens internal state, a simple check for existence.
            logger.info(f"GraPhens lookup embedding strategy configured.")
        elif embedding_config.get('type') == 'memmap':
            logger.info(f"GraPhens memmap data path: {embedding_config.get('memmap_data_path')}")
        else: # sentence_transformer, openai, etc.
            model_identifier = embedding_config.get('model_name') or embedding_config.get('model_name_or_path') or embedding_config.get('model')
            logger.info(f"GraPhens embedding model: {model_identifier}")

    except Exception as e:
        logger.error(f"Error initializing GraPhens: {e}")
        logger.error("Ensure HPO data (e.g., data/ontology/hp.json or hp.obo) and default embeddings are available.")
        return

    # --- 4. Create Graphs using GraPhens ---
    logger.info(f"Creating graphs for {len(patient_phenotypes_map)} samples using GraPhens...")
    try:
        dict_of_graphs = g.create_graphs_from_multiple_patients(patient_phenotypes_map, show_progress=True)
    except ValueError as ve: # Catch specific errors like "Phenotype list cannot be empty."
        logger.error(f"ValueError during graph creation in GraPhens: {ve}")
        logger.error("This might happen if a sample has an empty phenotype list that wasn't filtered, or an invalid HP ID.")
        return
    except Exception as e:
        logger.error(f"Unexpected error during graph creation in GraPhens: {e}")
        return
        
    logger.info(f"GraPhens processed {len(dict_of_graphs)} samples into internal Graph objects.")

    if not dict_of_graphs:
        logger.error("No graphs were successfully created by GraPhens. Exiting.")
        return

    # --- 5. Process and Collect PyG Data objects ---
    pyg_data_list = []
    
    # Create gene to label mapping
    all_gene_symbols_in_output = sorted(list(set(sample_to_gene_map[sid] for sid in dict_of_graphs.keys())))
    gene_to_label = {gene: i for i, gene in enumerate(all_gene_symbols_in_output)}
    
    logger.info(f"Created label mapping for {len(gene_to_label)} unique genes.")
    if len(gene_to_label) < 50: # Log small mappings
        logger.info(f"Gene-to-label map: {gene_to_label}")


    for sample_id, core_graph_object in dict_of_graphs.items():
        try:
            pyg_data_object = g.export_graph(core_graph_object, format="pytorch")
            gene_symbol = sample_to_gene_map[sample_id]
            
            pyg_data_object.sample_id = sample_id
            pyg_data_object.gene_symbol = gene_symbol
            pyg_data_object.y = torch.tensor([gene_to_label[gene_symbol]], dtype=torch.long)
            
            pyg_data_list.append(pyg_data_object)
        except Exception as e:
            logger.error(f"Error exporting graph or assigning label for sample {sample_id} (Gene: {sample_to_gene_map.get(sample_id, 'N/A')}): {e}")

    logger.info(f"Successfully created {len(pyg_data_list)} PyTorch Geometric Data objects.")

    if not pyg_data_list:
        logger.error("No PyTorch Geometric Data objects were created. Cannot save dataset.")
        return

    # --- 6. Save Dataset and Label Map ---
    try:
        torch.save(pyg_data_list, output_pt_path)
        logger.info(f"Successfully saved {len(pyg_data_list)} data objects to {output_pt_path}")
    except Exception as e:
        logger.error(f"Error saving PyTorch dataset: {e}")
        return

    try:
        with open(output_label_map_path, 'w') as f:
            json.dump(gene_to_label, f, indent=4)
        logger.info(f"Successfully saved gene-to-label mapping to {output_label_map_path}")
    except Exception as e:
        logger.error(f"Error saving gene-to-label mapping: {e}")

    logger.info("Dataset generation process finished.")

if __name__ == "__main__":
    # It's recommended to run this script from the project root directory
    # (e.g., /home/brainy/GraPhens) using:
    # python validation_arena/create_ddd_finetuning_set.py
    logger.info(f"Current working directory: {Path.cwd()}")
    logger.info("Ensure this script is run from the project root for GraPhens to find data directories (data/ontology, data/embeddings).")
    generate_finetuning_set() 