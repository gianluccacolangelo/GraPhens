#!/usr/bin/env python3
import json
import csv
import os
import argparse
import re
import torch
from pathlib import Path
import sys
import tqdm

# Add project root to sys.path to allow relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from training.models.models import GenePhenAIv2_0,GenePhenAIv2_5
from src.graphens import GraPhens
from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset

def load_phenotypes_from_json(json_path):
    """Load gene-phenotype mappings from JSON file."""
    print(f"Loading phenotypes data from JSON: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)

def load_phenotypes_from_tsv(tsv_path):
    """Load gene-phenotype mappings from TSV file with format: ID, Gene, Phenotype."""
    print(f"Loading phenotypes data from TSV: {tsv_path}")
    gene_to_phenotypes = {}
    
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene = row['Gene']
            # Convert comma-separated phenotype string to list
            phenotypes = row['Phenotype'].split(',')
            
            # Initialize gene in dictionary if not present
            if gene not in gene_to_phenotypes:
                gene_to_phenotypes[gene] = []
                
            # Add this phenotype set to the gene
            gene_to_phenotypes[gene].append(phenotypes)
            
    return gene_to_phenotypes

def main():
    parser = argparse.ArgumentParser(description="Validate GenePhenIA v2 model against a dataset of phenotypes and genes.")
    
    # Data input options - allow either JSON or TSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--phenotypes_json', type=str, 
                        help='Path to the JSON file with gene-phenotype mappings.')
    input_group.add_argument('--phenotypes_tsv', type=str,
                        help='Path to the TSV file with ID, Gene, Phenotype columns.')
    
    parser.add_argument('--output_csv', type=str, default='validation_results.csv',
                        help='Path to the output CSV file with validation results.')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/home/gcolangelo/GraPhens/training_output/checkpoint_epoch_1_part_2.pt',
                        help='Path to the model checkpoint.')
    parser.add_argument('--dataset_root', type=str, 
                        default='/home/gcolangelo/GraPhens/data/simulation/output/',
                        help='Path to the dataset root for model metadata.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of phenotype sets to process (for testing).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing phenotype sets.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu).')
    parser.add_argument('--embedding_model', type=str, default='gsarti/biobert-nli',
                        help='Name of the embedding model to use.')
    parser.add_argument('--hidden_channels', type=int, default=756,
                        help='Number of hidden units in GNN layers.')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata to get gene list and number of classes
    print(f"Loading metadata from dataset root: {args.dataset_root}")
    dataset_meta = LMDBHPOGraphDataset(root_dir=args.dataset_root, readonly=True)
    num_classes = dataset_meta.num_classes
    gene_list = dataset_meta.metadata.get('genes', [])
    idx_to_gene = {i: gene for i, gene in enumerate(gene_list)}
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = GenePhenAIv2_0(
        in_channels=768,  # Changed from 256 to match checkpoint error message (shape [128, 768])
        hidden_channels=756, # Changed from args.hidden_channels to match checkpoint error message (shape [128, ...])
        out_channels=num_classes
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize GraPhens (only once)
    print("Initializing GraPhens with settings matching training preprocessing...")
    try:
        graphens = (
            GraPhens()
            .with_lookup_embeddings("data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl")
            .with_augmentation(include_ancestors=True)
            .with_adjacency_settings(include_reverse_edges=False)
        )
        print("GraPhens initialized with lookup embeddings, ancestor augmentation, and forward-only edges.")
    except Exception as e:
        print(f"Error initializing GraPhens: {e}")
        sys.exit(1)
    
    # Load phenotypes data from either JSON or TSV
    if args.phenotypes_json:
        gene_to_phenotypes = load_phenotypes_from_json(args.phenotypes_json)
    else:
        gene_to_phenotypes = load_phenotypes_from_tsv(args.phenotypes_tsv)
    
    # Setup CSV output
    fieldnames = ['gene', 'phenotypes', 'score', 'rank']
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process genes
        total_samples = 0
        gene_count = 0
        
        # Process genes in batches to optimize
        all_phenotype_batches = []
        all_gene_labels = []
        all_phenotype_lists = []
        
        # Collect all phenotype sets to process
        for gene, phenotype_sets in gene_to_phenotypes.items():
            gene_count += 1
            print(f"Preparing gene {gene_count}/{len(gene_to_phenotypes)}: {gene}")
            
            for phenotypes in phenotype_sets:
                if not phenotypes:  # Skip empty phenotype sets
                    continue
                
                # Check if we've reached maximum samples
                total_samples += 1
                if args.max_samples and total_samples > args.max_samples:
                    break
                
                # Add to the batch list
                all_phenotype_batches.append(phenotypes)
                all_gene_labels.append(gene)
                all_phenotype_lists.append(' '.join(phenotypes))
            
            if args.max_samples and total_samples >= args.max_samples:
                break
        
        # Process in batches
        print(f"Processing {len(all_phenotype_batches)} phenotype sets in batches...")
        results = []
        
        for i in range(0, len(all_phenotype_batches), args.batch_size):
            batch_phenotypes = all_phenotype_batches[i:i+args.batch_size]
            batch_labels = all_gene_labels[i:i+args.batch_size]
            
            print(f"Processing batch {i//args.batch_size + 1}/{(len(all_phenotype_batches)-1)//args.batch_size + 1}")
            
            # Create a mapping from patient_id to phenotype sets for this batch
            batch_dict = {f"patient_{j}": phenos for j, phenos in enumerate(batch_phenotypes)}
            
            # Use batch processing to create graphs
            try:
                # Generate graphs for all patients in the batch
                graphs = graphens.create_graphs_from_multiple_patients(batch_dict, show_progress=True)
                
                # Process each graph with the model
                for j, (patient_id, graph) in enumerate(graphs.items()):
                    pyg_data = graphens.export_graph(graph, format="pytorch")
                    pyg_data = pyg_data.to(device)
                    
                    # Inference
                    with torch.no_grad():
                        output_logits = model(pyg_data.x, pyg_data.edge_index, pyg_data.batch)
                    
                    # Apply softmax to get probabilities for all classes
                    probabilities = torch.softmax(output_logits, dim=1)[0]
                    
                    # Get the true gene for this sample
                    true_gene = batch_labels[j]
                    gene_idx = None
                    
                    # Find gene index in the model's output
                    for idx, gene in idx_to_gene.items():
                        if gene == true_gene:
                            gene_idx = idx
                            break
                    
                    # Get the rank and score
                    if gene_idx is not None:
                        # Get the score for this gene
                        score = probabilities[gene_idx].item()
                        
                        # Compute exact rank by counting genes with higher probability
                        # +1 because ranks are 1-indexed
                        rank = (probabilities > probabilities[gene_idx]).sum().item() + 1
                    else:
                        # Gene not found in model's outputs
                        rank = "N/A"
                        score = 0.0
                    
                    # Write to CSV
                    writer.writerow({
                        'gene': true_gene,
                        'phenotypes': all_phenotype_lists[i+j],
                        'score': score,
                        'rank': rank
                    })
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Write errors to CSV
                for j in range(len(batch_phenotypes)):
                    writer.writerow({
                        'gene': batch_labels[j],
                        'phenotypes': all_phenotype_lists[i+j],
                        'score': "ERROR",
                        'rank': "ERROR"
                    })
    
    print(f"Validation completed. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()