#!/usr/bin/env python
"""Demo showing integration of simulation with GraPhens for GNN training."""
import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any, Tuple
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

# GraPhens imports
from src.simulation.phenotype_simulation.factory import SimulationFactory
from src.core.types import Phenotype, Graph
from src.graph.assembler import StandardGraphAssembler
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import SentenceTransformerEmbeddingStrategy, LookupEmbeddingStrategy
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('integration_demo')


class SimpleGNN(torch.nn.Module):
    """Simple Graph Neural Network for demonstration purposes."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 16):
        """Initialize the GNN.
        
        Args:
            input_dim: Input dimension (node feature dimension)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super(SimpleGNN, self).__init__()
        
        # Import here to avoid dependency if not using GNN
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.pool = global_mean_pool
    
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Node embeddings and graph embedding
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create a batch with all nodes in one graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pooling to get graph representation
        graph_embedding = self.pool(x, batch)
        
        return x, graph_embedding


def simulate_patients(args: argparse.Namespace) -> Dict[str, List[List[Phenotype]]]:
    """Simulate patient phenotypes using the phenotype simulation module.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary mapping gene symbols to lists of patient phenotype lists
    """
    logger.info("1. Simulating patient phenotypes")
    
    # Create the simulator
    simulator = SimulationFactory.create_complete_simulator(
        data_dir_simulation=args.data_dir_simulation,
        data_dir_hpo=args.data_dir_hpo,
        data_dir_gene_phenotype=args.data_dir_gene_phenotype
    )
    
    # Prepare gene-to-count mapping
    gene_to_count = {}
    if args.genes:
        # Use specified genes
        for gene in args.genes:
            gene_to_count[gene] = args.patients_per_gene
    else:
        # Use genes from JSON file
        with open(args.gene_file, 'r') as f:
            genes = json.load(f)
            for gene in genes:
                gene_to_count[gene] = args.patients_per_gene
    
    # Simulate patients
    logger.info(f"Simulating {sum(gene_to_count.values())} patients for {len(gene_to_count)} genes")
    gene_patients = simulator.generate_patients_with_tqdm(gene_to_count)
    
    return gene_patients


def convert_to_graphs(
    gene_patients: Dict[str, List[List[Phenotype]]],
    args: argparse.Namespace
) -> Tuple[List[Graph], List[str]]:
    """Convert simulated phenotypes to graph representations.
    
    Args:
        gene_patients: Dictionary mapping gene symbols to lists of patient phenotype lists
        args: Command-line arguments
        
    Returns:
        Tuple of (list of Graph objects, list of corresponding gene labels)
    """
    logger.info("2. Converting phenotypes to graph representations")
    
    # Initialize components
    hpo_provider = HPOGraphProvider.get_instance(data_dir=args.data_dir_hpo)
    
    # Initialize graph components
    adjacency_builder = HPOAdjacencyListBuilder(
        hpo_provider=hpo_provider,
        include_reverse_edges=True
    )
    
    graph_assembler = StandardGraphAssembler()
    
    # Initialize embedding strategy
    if args.embeddings_file:
        # Use pre-computed embeddings
        import pickle
        with open(args.embeddings_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        embedding_strategy = LookupEmbeddingStrategy(embedding_dict=embeddings_dict)
        logger.info(f"Using pre-computed embeddings from {args.embeddings_file}")
    else:
        # Use SentenceTransformer
        embedding_strategy = SentenceTransformerEmbeddingStrategy(
            model_name=args.embedding_model
        )
        logger.info(f"Using SentenceTransformer model: {args.embedding_model}")
    
    embedding_context = EmbeddingContext(embedding_strategy)
    
    # Create augmentation service if needed
    if args.augment:
        augmentation_service = HPOAugmentationService(
            data_dir=args.data_dir_hpo,
            include_ancestors=args.include_ancestors,
            include_descendants=args.include_descendants
        )
        logger.info(f"Using augmentation (ancestors={args.include_ancestors}, descendants={args.include_descendants})")
    
    # Process each gene's patients
    graphs = []
    labels = []
    
    for gene, patients in tqdm(gene_patients.items(), desc="Processing genes"):
        for i, phenotypes in enumerate(patients):
            # Augment phenotypes if needed
            if args.augment:
                phenotypes = augmentation_service.augment(phenotypes)
            
            # Create embeddings
            node_features = embedding_context.embed_phenotypes(phenotypes)
            
            # Build adjacency list (edges)
            edge_index = adjacency_builder.build(phenotypes)
            
            # Assemble graph
            graph = graph_assembler.assemble(
                phenotypes=phenotypes,
                node_features=node_features,
                edge_index=edge_index,
                metadata={
                    "gene": gene,
                    "patient_id": f"{gene}_{i}",
                    "phenotypes": phenotypes
                }
            )
            
            graphs.append(graph)
            labels.append(gene)
    
    logger.info(f"Created {len(graphs)} graph representations from {len(gene_patients)} genes")
    return graphs, labels


def export_to_pytorch(
    graphs: List[Graph],
    labels: List[str],
    args: argparse.Namespace
) -> Tuple[List, List[int]]:
    """Export graphs to PyTorch Geometric format.
    
    Args:
        graphs: List of Graph objects
        labels: List of corresponding gene labels
        args: Command-line arguments
        
    Returns:
        Tuple of (list of PyTorch Geometric Data objects, list of numeric labels)
    """
    logger.info("3. Exporting graphs to PyTorch Geometric format")
    
    # Import PyTorch Geometric
    from torch_geometric.data import Data
    
    # Convert gene labels to numeric labels
    unique_genes = sorted(set(labels))
    gene_to_idx = {gene: i for i, gene in enumerate(unique_genes)}
    numeric_labels = [gene_to_idx[gene] for gene in labels]
    
    # Save mapping
    mapping_file = os.path.join(args.output_dir, "gene_label_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(gene_to_idx, f, indent=2)
    logger.info(f"Saved gene label mapping to {mapping_file}")
    
    # Convert graphs to PyTorch Geometric Data objects
    pyg_graphs = []
    for i, (graph, label) in enumerate(zip(graphs, numeric_labels)):
        # Convert to PyTorch tensors
        x = torch.tensor(graph.node_features, dtype=torch.float)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        y = torch.tensor([label], dtype=torch.long)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            patient_id=graph.metadata.get("patient_id", f"patient_{i}")
        )
        
        pyg_graphs.append(data)
    
    logger.info(f"Converted {len(pyg_graphs)} graphs to PyTorch Geometric format")
    return pyg_graphs, numeric_labels


def train_simple_model(
    pyg_graphs: List,
    numeric_labels: List[int],
    args: argparse.Namespace
) -> None:
    """Train a simple GNN model on the synthetic data.
    
    Args:
        pyg_graphs: List of PyTorch Geometric Data objects
        numeric_labels: List of numeric labels
        args: Command-line arguments
    """
    logger.info("4. Training a simple GNN model")
    
    # Split data into train/validation
    np.random.seed(42)
    indices = np.random.permutation(len(pyg_graphs))
    train_idx = indices[:int(0.8 * len(indices))]
    val_idx = indices[int(0.8 * len(indices)):]
    
    train_graphs = [pyg_graphs[i] for i in train_idx]
    val_graphs = [pyg_graphs[i] for i in val_idx]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    
    # Determine input dimension from the data
    input_dim = pyg_graphs[0].x.size(1)
    
    # Create model
    model = SimpleGNN(input_dim=input_dim)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train the model
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            optimizer.zero_grad()
            _, graph_embedding = model(data)
            loss = criterion(graph_embedding, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            pred = graph_embedding.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in val_loader:
                _, graph_embedding = model(data)
                loss = criterion(graph_embedding, data.y)
                
                val_loss += loss.item()
                
                # Compute accuracy
                pred = graph_embedding.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                   f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Save model
    model_file = os.path.join(args.output_dir, "simple_gnn_model.pt")
    torch.save(model.state_dict(), model_file)
    logger.info(f"Saved trained model to {model_file}")


def save_pytorch_data(
    pyg_graphs: List,
    numeric_labels: List[int],
    args: argparse.Namespace
) -> None:
    """Save PyTorch Geometric data for later use.
    
    Args:
        pyg_graphs: List of PyTorch Geometric Data objects
        numeric_labels: List of numeric labels
        args: Command-line arguments
    """
    logger.info("5. Saving PyTorch Geometric data")
    
    # Save as pickle file
    import pickle
    data_file = os.path.join(args.output_dir, "pyg_graphs.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(pyg_graphs, f)
    
    # Save a summary file
    summary = {
        "num_graphs": len(pyg_graphs),
        "num_classes": len(set(numeric_labels)),
        "class_distribution": {i: numeric_labels.count(i) for i in set(numeric_labels)},
        "node_feature_dim": pyg_graphs[0].x.size(1) if pyg_graphs else None,
    }
    
    summary_file = os.path.join(args.output_dir, "pytorch_data_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved PyTorch Geometric data to {data_file}")
    logger.info(f"Saved data summary to {summary_file}")


def main():
    """Run the integration demo."""
    parser = argparse.ArgumentParser(description="Phenotype Simulation GNN Integration Demo")
    
    # Directory arguments
    parser.add_argument("--data-dir-simulation", type=str, default="data/simulation",
                        help="Directory for simulation data files")
    parser.add_argument("--data-dir-hpo", type=str, default="data/ontology",
                        help="Directory for HPO data")
    parser.add_argument("--data-dir-gene-phenotype", type=str, default="data/gene_phenotype",
                        help="Directory for gene-phenotype data")
    parser.add_argument("--output-dir", type=str, default="data/simulation/output",
                        help="Directory for output files")
    
    # Simulation arguments
    parser.add_argument("--genes", type=str, nargs="+",
                        help="Gene symbols to simulate patients for")
    parser.add_argument("--gene-file", type=str,
                        help="JSON file containing list of genes to simulate")
    parser.add_argument("--patients-per-gene", type=int, default=100,
                        help="Number of patients to simulate per gene")
    
    # Graph conversion arguments
    parser.add_argument("--augment", action="store_true",
                        help="Whether to augment phenotypes")
    parser.add_argument("--include-ancestors", action="store_true", default=True,
                        help="Include ancestor terms in augmentation")
    parser.add_argument("--include-descendants", action="store_true",
                        help="Include descendant terms in augmentation")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model to use for embeddings")
    parser.add_argument("--embeddings-file", type=str,
                        help="File containing pre-computed embeddings (pickle)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true",
                        help="Train a simple GNN model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run the pipeline
    try:
        # Step 1: Simulate patients
        gene_patients = simulate_patients(args)
        
        # Step 2: Convert to graph representations
        graphs, labels = convert_to_graphs(gene_patients, args)
        
        # Step 3: Export to PyTorch Geometric format
        pyg_graphs, numeric_labels = export_to_pytorch(graphs, labels, args)
        
        # Step 4: Save PyTorch data
        save_pytorch_data(pyg_graphs, numeric_labels, args)
        
        # Step 5: Train a simple model (optional)
        if args.train:
            train_simple_model(pyg_graphs, numeric_labels, args)
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in integration demo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 