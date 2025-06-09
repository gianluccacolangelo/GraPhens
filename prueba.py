from src.graphens import GraPhens

# Initialize with configuration
graphens = GraPhens(config_path="config/default.yaml")

# Customize components as needed
graphens.with_memmap_embeddings()
graphens.with_augmentation(include_ancestors=True)
graphens.with_visualization(output_dir="output/visualizations")

# Process multiple patients
patient_data = {
    "patient_1": ["HP:0001250", "HP:0002066", "HP:0100543"],
    "patient_2": ["HP:0000407", "HP:0001263", "HP:0001290"],
    "patient_3": ["HP:0000256", "HP:0000486", "HP:0000556"]
}

# Create graphs efficiently
patient_graphs = graphens.create_graphs_from_multiple_patients(patient_data)

# Option 1: Export as a dictionary of individual graphs
individual_pyg_graphs = graphens.export_graph(patient_graphs, format='pytorch', batch=False)
print("Individual graphs:", individual_pyg_graphs)

# Option 2: Export as a single batched graph (more efficient for training)
batched_pyg_graph = graphens.export_graph(patient_graphs, format='pytorch', batch=True)
print('='*100)
print(f'\nBatched graph for PyTorch:\n{batched_pyg_graph}')

# Access individual graphs from the batch if needed
if batched_pyg_graph is not None:
    print(f'\nNumber of graphs in batch: {batched_pyg_graph.num_graphs}')
    print(f'Patient IDs in batch: {batched_pyg_graph.patient_ids}')
    
    # Print first few entries from node_mappings for each patient
    print("\nSample node mappings:")
    for patient_id in batched_pyg_graph.patient_ids:
        node_mapping = batched_pyg_graph.node_mappings[patient_id]
        # Print first 3 items or fewer if mapping is smaller
        items = list(node_mapping.items())[:3]
        print(f"  {patient_id}: {items}")

look_for_phen = graphens.phenotype_lookup("broad forehead")
print(look_for_phen)