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

from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5, GenePhenAIv3_0
from src.graphens import GraPhens

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
    parser.add_argument('--model_version', type=str, choices=['2.0', '2.5', '3.0'],
                        help='Model version to use. Loaded from checkpoint args.json if available.')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads for v2.5. Loaded from checkpoint args.json if available.')
    
    args = parser.parse_args()
    
    # --- Load Training Arguments and Override ---
    training_args = {}
    checkpoint_dir = Path(args.checkpoint_path).parent
    args_json_path = checkpoint_dir / 'args.json'

    if args_json_path.is_file():
        print(f"Found training arguments file: {args_json_path}")
        with open(args_json_path, 'r') as f:
            training_args = json.load(f)

        # Override args with values from training_args
        args.model_version = training_args.get('model_version', args.model_version)
        args.hidden_channels = training_args.get('hidden_channels', args.hidden_channels)
        args.num_heads = training_args.get('num_heads', args.num_heads)
        
        print(f"Loaded from args.json: model_version='{args.model_version}', hidden_channels={args.hidden_channels}, num_heads={args.num_heads}")
    else:
        print(f"Warning: Training arguments file not found at {args_json_path}. Will rely on command-line arguments for model hyperparameters.")

    # Set up device
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load metadata to get gene list and number of classes
    print(f"Loading metadata from dataset root: {args.dataset_root}")
    try:
        metadata_path = Path(args.dataset_root) / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        gene_list = metadata.get('genes', [])
        if not gene_list:
            print("Could not retrieve gene list from dataset metadata. Ensure 'genes' key exists in metadata.json.")
            sys.exit(1)
        
        num_classes = len(gene_list)
        idx_to_gene = {i: gene for i, gene in enumerate(gene_list)}
        print(f"Successfully loaded metadata: Found {num_classes} classes (genes).")

    except FileNotFoundError as e:
        print(f"{e}. Please provide the correct path to the directory containing 'metadata.json'.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {metadata_path}. The file may be corrupt.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset metadata: {e}")
        sys.exit(1)
    
    # --- Instantiate Model ---
    print("Instantiating model...")

    # Check if necessary model parameters are available
    if not args.model_version:
        print("Model version not specified. Provide --model_version or a checkpoint with args.json.")
        sys.exit(1)
    if not args.hidden_channels:
        print("Hidden channels not specified. Provide --hidden_channels or a checkpoint with args.json.")
        sys.exit(1)

    model_params = {
        'in_channels': 768,
        'hidden_channels': args.hidden_channels,
        'out_channels': num_classes
    }

    model_class = None
    if args.model_version == '2.5':
        if not args.num_heads:
            print("Number of heads (--num_heads) is required for model_version 2.5 and was not found in args.json.")
            sys.exit(1)
        model_class = GenePhenAIv2_5
        model_params['num_heads'] = args.num_heads
        print(f"Instantiating GenePhenAIv2_5 model with params: {model_params}")
    elif args.model_version == '3.0':
        model_class = GenePhenAIv3_0
        print(f"Instantiating GenePhenAIv3_0 model with params: {model_params}")
    elif args.model_version == '2.0':
        model_class = GenePhenAIv2_0
        print(f"Instantiating GenePhenAIv2_0 model with params: {model_params}")
    else:
        print(f"Unsupported model version: {args.model_version}. Cannot instantiate model.")
        sys.exit(1)

    try:
        model = model_class(**model_params).to(device)
    except Exception as e:
        print(f"Error instantiating model: {e}")
        sys.exit(1)

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {args.checkpoint_path}")
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