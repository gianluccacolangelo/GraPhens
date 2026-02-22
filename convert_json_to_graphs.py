import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import entropy
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.core.types import Phenotype, Graph
from src.graph.assembler import StandardGraphAssembler
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import LookupEmbeddingStrategy
from src.ontology.hpo_graph import HPOGraphProvider
from src.augmentation.hpo_augmentation import HPOAugmentationService

# Try to import UMAP, fallback to TSNE if not available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

def load_phenotypes(phenotype_ids: List[str], hpo_provider: HPOGraphProvider) -> List[Phenotype]:
    phenotypes = []
    for hpo_id in phenotype_ids:
        norm_id = hpo_provider._normalize_hpo_id(hpo_id)
        metadata = hpo_provider.get_metadata(norm_id)
        if not metadata:
            name = norm_id
            description = None
        else:
            name = metadata.get('name', norm_id)
            description = metadata.get('definition', None)
        
        phenotypes.append(Phenotype(
            id=norm_id,
            name=name,
            description=description
        ))
    return phenotypes

def get_root_categories(hpo_provider: HPOGraphProvider) -> Dict[str, str]:
    """
    Get the direct children of Phenotypic Abnormality (HP:0000118) to use as high-level categories.
    """
    root_id = "HP:0000118"
    children = hpo_provider.get_direct_children(root_id)
    
    categories = {}
    for child_id in children:
        metadata = hpo_provider.get_metadata(child_id)
        name = metadata.get('name', child_id)
        categories[child_id] = name
        
    return categories

def get_dominant_category(phenotypes: List[Phenotype], categories: Dict[str, str], hpo_provider: HPOGraphProvider) -> Tuple[str, str]:
    """
    Determine the dominant category for a list of phenotypes based on their ancestors.
    Returns (category_id, category_name).
    """
    category_counts = Counter()
    
    for p in phenotypes:
        ancestors = hpo_provider.get_ancestors(p.id)
        # Check which category ID is in the ancestors
        found_category = False
        for cat_id in categories:
            if cat_id in ancestors:
                category_counts[cat_id] += 1
                found_category = True
        
        # If no specific category found (e.g. it's the root itself or weird), maybe map to Unknown
        if not found_category:
            category_counts["Unknown"] += 1

    if not category_counts:
        return "Unknown", "Unknown"
        
    best_cat_id, _ = category_counts.most_common(1)[0]
    return best_cat_id, categories.get(best_cat_id, "Unknown")

def calculate_graph_complexity(graph: Graph, phenotypes: List[Phenotype]) -> Dict[str, float]:
    """
    Calculate complexity metrics for the graph: Depth, Width, Entropy.
    """
    # Convert to NetworkX for easy analysis
    # The edge_index is [2, num_edges] where row 0 is source, row 1 is target
    edges = graph.edge_index.T
    G = nx.DiGraph()
    
    # Add nodes
    for i, p in enumerate(phenotypes):
        G.add_node(i, id=p.id)
        
    # Add edges
    for src, dst in edges:
        G.add_edge(src, dst)
        
    # 1. Entropy (Shannon entropy of degree distribution)
    degrees = [d for n, d in G.degree()]
    if not degrees:
        graph_entropy = 0
    else:
        total_degree = sum(degrees)
        if total_degree == 0:
             graph_entropy = 0
        else:
            probs = [d / total_degree for d in degrees]
            graph_entropy = entropy(probs)

    # 2. Depth (Longest path in DAG)
    try:
        if nx.is_directed_acyclic_graph(G):
            depth = nx.dag_longest_path_length(G)
        else:
            # Fallback for cyclic graphs (shouldn't happen with include_reverse_edges=False)
            depth = 0
    except Exception:
        depth = 0

    # 3. Width (Max nodes at any topological level)
    # We can use topological generations
    try:
        generations = list(nx.topological_generations(G))
        width = max(len(gen) for gen in generations) if generations else 0
    except Exception:
        width = 0

    return {
        "depth": depth,
        "width": width,
        "entropy": graph_entropy,
        "density": nx.density(G),
        "cyclomatic_complexity": graph.edge_index.shape[1] - graph.node_features.shape[0] + nx.number_weakly_connected_components(G)
    }

def analyze_and_plot(df: pd.DataFrame, embeddings: np.ndarray, output_dir: str):
    print("\n--- Deep Data Analysis ---")
    print(f"Total Graphs: {len(df)}")
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, 'graph_analysis_full.csv'), index=False)
    
    # 1. Complexity Distributions
    metrics = ['num_nodes', 'num_edges', 'depth', 'width', 'entropy', 'density']
    # Set fixed limits based on the maximum range observed across datasets (Original vs Simulated)
    # Original max: nodes~262, edges~305, depth~15, width~61, entropy~5.45
    # Simulated max: nodes~95, edges~124, depth~15, width~19, entropy~4.5
    limits = {
        'num_nodes': (0, 300),
        'num_edges': (0, 350),
        'depth': (0, 16),
        'width': (0, 70),
        'entropy': (0, 6.0),
        'density': (0, 0.6)
    }
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[metric], kde=True)
        if metric in limits:
            plt.xlim(limits[metric])
        plt.title(f'{metric.replace("_", " ").title()} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_distributions.png'))
    plt.close()

    # 1.5 Augmentation Statistics (Original vs Augmented)
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Original Phenotype Count Distribution
    plt.subplot(1, 3, 1)
    sns.histplot(df['original_count'], bins=range(int(df['original_count'].min()), int(df['original_count'].max()) + 2), kde=False)
    plt.title('Original Phenotype Count Distribution')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    
    # Plot 2: Augmented Node Count Distribution
    plt.subplot(1, 3, 2)
    sns.histplot(df['augmented_count'], kde=True, color='orange')
    plt.title('Augmented Node Count Distribution')
    plt.xlabel('Count')
    
    # Plot 3: Scatter Original vs Augmented
    plt.subplot(1, 3, 3)
    plt.scatter(df['original_count'], df['augmented_count'], alpha=0.5)
    plt.title('Original vs Augmented Count')
    plt.xlabel('Original')
    plt.ylabel('Augmented')
    
    # Add trend line
    if len(df) > 1:
        try:
            z = np.polyfit(df['original_count'], df['augmented_count'], 1)
            p = np.poly1d(z)
            plt.plot(df['original_count'], p(df['original_count']), "r--", alpha=0.8, label=f"y={z[0]:.2f}x+{z[1]:.2f}")
            plt.legend()
        except Exception:
            pass
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phenotype_augmentation_stats.png'))
    plt.close()

    # 2. UMAP Projection
    print("\nCalculating Projection...")
    if HAS_UMAP:
        print("Using UMAP...")
        reducer = umap.UMAP(n_components=2, n_neighbors=170,metric='cosine')
    else:
        print("UMAP not found, using t-SNE...")
        reducer = TSNE(n_components=2, random_state=42)
        
    # Ensure embeddings are valid
    if np.isnan(embeddings).any():
        print("Warning: NaNs in embeddings, replacing with 0")
        embeddings = np.nan_to_num(embeddings)
        
    projection = reducer.fit_transform(embeddings)
    
    # Add projection to DF
    df['x'] = projection[:, 0]
    df['y'] = projection[:, 1]
    
    # Plot by Category
    plt.figure(figsize=(16, 12))
    
    # Get top categories to avoid clutter if too many
    top_cats = df['category_name'].value_counts().head(20).index
    plot_df = df[df['category_name'].isin(top_cats)].copy()
    
    # Sort categories by count for legend order
    hue_order = df['category_name'].value_counts().index[:20]
    
    sns.scatterplot(
        data=plot_df, 
        x='x', y='y', 
        hue='category_name', 
        alpha=0.6, 
        palette='tab20',
        hue_order=hue_order,
        s=15
    )
    plt.title(f'Graph Embeddings Projection ({("UMAP" if HAS_UMAP else "t-SNE")}) by Top 20 HPO Categories')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Primary Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_projection_by_category.png'))
    plt.close()
    
    # Plot by Complexity (Depth)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, 
        x='x', y='y', 
        hue='depth', 
        palette='viridis', 
        alpha=0.6,
        s=15
    )
    plt.title('Graph Embeddings Projection - Colored by Graph Depth')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_projection_by_depth.png'))
    plt.close()

    # Plot by Complexity (Entropy)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, 
        x='x', y='y', 
        hue='entropy', 
        palette='magma', 
        alpha=0.6,
        s=15
    )
    plt.title('Graph Embeddings Projection - Colored by Graph Entropy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'graph_projection_by_entropy.png'))
    plt.close()
    
    print(f"Analysis complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deep Graph Analysis")
    parser.add_argument('--input-file', type=str, default="src/simulation/fenotipos_cleaned_processed.json", help="Input JSON file")
    parser.add_argument('--embeddings-file', type=str, default="data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl", help="Path to embedding pickle file")
    parser.add_argument('--ontology-dir', type=str, default="data/ontology", help="Ontology directory")
    parser.add_argument('--output-dir', type=str, default="output/analysis", help="Output directory")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of genes to process")
    parser.add_argument('--min-slope', type=float, default=None, help="Filter out points with augmented/original ratio below this value")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Resources
    print(f"Loading HPO Provider from {args.ontology_dir}...")
    hpo_provider = HPOGraphProvider.get_instance(data_dir=args.ontology_dir)
    
    print("Getting Root Categories (Level 3)...")
    root_categories = get_root_categories(hpo_provider)
    print(f"Found {len(root_categories)} categories: {list(root_categories.values())[:5]}...")
    
    print(f"Loading Embeddings from {args.embeddings_file}...")
    with open(args.embeddings_file, 'rb') as f:
        embedding_dict = pickle.load(f)
    
    if isinstance(embedding_dict, tuple):
        embedding_dict = embedding_dict[0]
        
    sample_key = next(iter(embedding_dict))
    dim = len(embedding_dict[sample_key])
    print(f"Embedding dimension: {dim}")
    
    embedding_strategy = LookupEmbeddingStrategy(embedding_dict, dim=dim)
    embedding_context = EmbeddingContext(embedding_strategy)
    
    augmentation_service = HPOAugmentationService(data_dir=args.ontology_dir, include_ancestors=True)
    # Disable reverse edges to ensure DAG property for depth calculation
    adjacency_builder = HPOAdjacencyListBuilder(hpo_provider, include_reverse_edges=False)
    graph_assembler = StandardGraphAssembler()
    
    # 2. Load Data
    print(f"Loading input JSON from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    genes = list(data.keys())
    if args.limit:
        genes = genes[:args.limit]
        print(f"Limiting to {args.limit} genes")
        
    results = []
    graph_embeddings = []
    
    # 3. Process
    for gene in tqdm(genes, desc="Processing Genes"):
        cases = data[gene]
        for idx, phenotype_ids in enumerate(cases):
            try:
                initial_phenotypes = load_phenotypes(phenotype_ids, hpo_provider)
                if not initial_phenotypes:
                    continue
                
                augmented_phenotypes = augmentation_service.augment(initial_phenotypes)
                node_features = embedding_context.embed_phenotypes(augmented_phenotypes)
                edge_index = adjacency_builder.build(augmented_phenotypes)
                graph = graph_assembler.assemble(augmented_phenotypes, node_features, edge_index)
                
                # Metrics
                complexity = calculate_graph_complexity(graph, augmented_phenotypes)
                cat_id, cat_name = get_dominant_category(augmented_phenotypes, root_categories, hpo_provider)
                
                # Graph Embedding (Mean Pooling)
                if graph.node_features.shape[0] > 0:
                    g_embed = np.mean(graph.node_features, axis=0)
                else:
                    g_embed = np.zeros(dim)
                
                graph_embeddings.append(g_embed)
                
                results.append({
                    'gene': gene,
                    'case_idx': idx,
                    'original_count': len(initial_phenotypes),
                    'augmented_count': len(augmented_phenotypes),
                    'num_nodes': graph.node_features.shape[0],
                    'num_edges': graph.edge_index.shape[1],
                    'category_id': cat_id,
                    'category_name': cat_name,
                    **complexity
                })
                
            except Exception as e:
                # print(f"Error: {e}")
                continue
                
    # 4. Analyze
    df = pd.DataFrame(results)
    embeddings_array = np.array(graph_embeddings)
    
    # Apply filtering consistently to both df and embeddings
    if args.min_slope is not None and len(df) > 0:
        # Calculate ratio
        ratios = df['augmented_count'] / df['original_count'].replace(0, 1)
        mask = ratios >= args.min_slope
        
        removed_count = (~mask).sum()
        print(f"Filtered out {removed_count} cases ({removed_count/len(df)*100:.1f}%) below slope {args.min_slope}")
        
        df = df[mask].reset_index(drop=True)
        if len(embeddings_array) > 0:
            embeddings_array = embeddings_array[mask]
    
    analyze_and_plot(df, embeddings_array, args.output_dir)

if __name__ == "__main__":
    main()
