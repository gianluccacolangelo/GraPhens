import argparse
import logging
import torch
import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from training.models.models import GenePhenAIv2_0, GenePhenAIv2_5, GenePhenAIv3_0
from src.graphens import GraPhens
from src.ontology.hpo_graph import HPOGraphProvider
from src.core.types import Phenotype
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.ontology.orphanet_mapper import OrphanetMapper

# Try to import UMAP, fallback to TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

from sklearn.metrics import silhouette_score, davies_bouldin_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActivationHook:
    def __init__(self):
        self.activation = None

    def __call__(self, module, input, output):
        self.activation = output

TARGET_CATEGORIES = {
    "ORPHA:98053": "Enfermedades genéticas raras",
    "ORPHA:91378": "Angioedema hereditario",
    "ORPHA:447874": "Anomalías biológicas sin caracterización fenotípica",
    "ORPHA:68335": "Anomalías cromosómicas raras",
    "ORPHA:156619": "Anomalías urogenitales genéticas raras",
    "ORPHA:363250": "Ciliopatías",
    "ORPHA:98054": "Enfermedades cardíacas genéticas raras",
    "ORPHA:68346": "Enfermedades de la piel genéticas raras",
    "ORPHA:156638": "Enfermedades endocrinas genéticas raras",
    "ORPHA:165652": "Enfermedades gastroenterológicas genéticas raras",
    "ORPHA:183731": "Enfermedades gineco-obstétricas genéticas raras",
    "ORPHA:158300": "Enfermedades hematológicas genéticas raras",
    "ORPHA:156601": "Enfermedades hepáticas genéticas raras",
    "ORPHA:183770": "Enfermedades inmunológicas genéticas raras",
    "ORPHA:71859": "Enfermedades neurológicas genéticas raras",
    "ORPHA:101435": "Enfermedades oculares genéticas raras",
    "ORPHA:77830": "Enfermedades odontológicas genéticas raras",
    "ORPHA:466084": "Enfermedades otorrinolaringológicas genéticas",
    "ORPHA:98056": "Enfermedades renales genéticas raras",
    "ORPHA:156610": "Enfermedades respiratorias genéticas raras",
    "ORPHA:271870": "Enfermedades sistémicas genéticas raras",
    "ORPHA:233655": "Enfermedades vasculares genéticas raras",
    "ORPHA:183524": "Enfermedades óseas genéticas raras",
    "ORPHA:68367": "Errores congénitos del metabolismo raros",
    "ORPHA:275742": "Infertilidad genética",
    "ORPHA:98301": "Laminopatías",
    "ORPHA:536391": "RASopatías",
    "ORPHA:250805": "Serpinopatías",
    "ORPHA:100974": "Síndrome FRAXF",
    "ORPHA:140162": "Síndromes de predisposición hereditaria al cáncer",
    "ORPHA:641343": "Trastornos de la impronta",
    "ORPHA:183530": "Trastornos del desarrollo embrionario genéticos raros",
    "ORPHA:68336": "Tumor genético raro"
}

def main():
    parser = argparse.ArgumentParser(description="Analyze GNN Latent Space Evolution")
    parser.add_argument('--input_file', type=str, default='src/simulation/fenotipos_cleaned_processed.json')
    parser.add_argument('--checkpoint_path', type=str, default='training_output/best_model.pt')
    parser.add_argument('--embedding_path', type=str, default='data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl')
    parser.add_argument('--ontology_dir', type=str, default='data/ontology')
    parser.add_argument('--output_dir', type=str, default='output/analysis/gnn_latent_space')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference (graphs per batch)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load HPO and Orphanet Mapper
    logger.info(f"Loading HPO from {args.ontology_dir}")
    # hpo_provider needed by GraPhens internally
    
    # Initialize Orphanet Mapper
    logger.info("Initializing Orphanet Mapper...")
    # Assuming files are in data/ relative to ontology_dir (data/ontology)
    # data/ordo_orphanet.owl and data/genes_to_phenotype.txt
    data_root = Path(args.ontology_dir).parent
    orphanet_mapper = OrphanetMapper(
        ordo_path=str(data_root / 'ordo_orphanet.owl'),
        gene_phenotype_path=str(data_root / 'genes_to_phenotype.txt'),
        root_categories=TARGET_CATEGORIES
    )

    # 2. Load Model
    logger.info(f"Loading model from {args.checkpoint_path}")
    checkpoint_dir = Path(args.checkpoint_path).parent
    args_json_path = checkpoint_dir / 'args.json'
    
    with open(args_json_path, 'r') as f:
        train_args = json.load(f)
        
    # Load Metadata for num_classes
    metadata_path = Path('data/simulation/output/metadata.json') 
    if not metadata_path.exists():
         metadata_path = Path('data/metadata.json')

    if metadata_path.exists():
         with open(metadata_path, 'r') as f:
            meta = json.load(f)
            num_classes = len(meta.get('genes', []))
    else:
        logger.warning("Metadata not found. Using dummy out_channels=2233 (from previous info).")
        num_classes = 2233

    embedding_dim = 768 # Default for BioBERT
    
    if train_args['model_version'] == '2.0':
        model = GenePhenAIv2_0(embedding_dim, train_args['hidden_channels'], num_classes)
    elif train_args['model_version'] == '2.5':
        model = GenePhenAIv2_5(embedding_dim, train_args['hidden_channels'], num_classes, num_heads=train_args.get('num_heads', 4))
    elif train_args['model_version'] == '3.0':
        model = GenePhenAIv3_0(embedding_dim, train_args['hidden_channels'], num_classes)
        
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 3. Register Hooks
    hooks = {
        'conv1': ActivationHook(),
        'conv2': ActivationHook(),
        'conv3': ActivationHook()
    }
    
    model.conv1.register_forward_hook(hooks['conv1'])
    model.conv2.register_forward_hook(hooks['conv2'])
    model.conv3.register_forward_hook(hooks['conv3'])

    # 4. Initialize GraPhens
    graphens = (
        GraPhens()
        .with_lookup_embeddings(args.embedding_path)
        .with_augmentation(include_ancestors=True)
        .with_adjacency_settings(include_reverse_edges=False)
    )

    # 5. Process Data
    logger.info(f"Loading data from {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    genes = list(data.keys())
    if args.limit:
        genes = genes[:args.limit]
    
    results = {
        'original': [],
        'step1': [],
        'step2': [],
        'step3': [],
        'category': []
    }
    
    count = 0
    for gene in tqdm(genes, desc="Processing Genes"):
        # Get Category from Gene (Orphanet)
        category = orphanet_mapper.get_category_for_gene(gene)
        
        cases = data[gene]
        for phenotype_ids in cases:
            try:
                # 2. Create Graph
                internal_graph = graphens.create_graph_from_phenotypes(phenotype_ids)
                if internal_graph.node_features.shape[0] == 0:
                    continue
                    
                # 3. Original Embedding (Mean Pool)
                orig_emb = np.mean(internal_graph.node_features, axis=0)
                
                # 4. Run Inference
                pyg_data = graphens.export_graph(internal_graph, format="pytorch").to(device)
                
                with torch.no_grad():
                    _ = model(pyg_data.x, pyg_data.edge_index, pyg_data.batch)
                
                # 5. Collect Latent Reps
                def pool_hook_output(hook_out):
                    return hook_out.mean(dim=0).cpu().numpy()

                step1_emb = pool_hook_output(hooks['conv1'].activation)
                step2_emb = pool_hook_output(hooks['conv2'].activation)
                step3_emb = pool_hook_output(hooks['conv3'].activation)
                
                results['original'].append(orig_emb)
                results['step1'].append(step1_emb)
                results['step2'].append(step2_emb)
                results['step3'].append(step3_emb)
                results['category'].append(category)
                
                count += 1
                
            except Exception as e:
                logger.error(f"Error processing case: {e}")
                continue

    # 6. UMAP & Plot
    logger.info(f"Collected {count} samples. Running Projection...")
    
    categories = np.array(results['category'])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    titles = ['Original Input (BioBERT)', 'Step 1 (GNN Layer 1)', 'Step 2 (GNN Layer 2)', 'Step 3 (GNN Layer 3)']
    keys = ['original', 'step1', 'step2', 'step3']
    
    metrics_results = []

    # Get top categories for legend
    top_cats = pd.Series(categories).value_counts().head(15).index.tolist()
    
    # Fix colors for categories across plots
    unique_cats = sorted(list(set(categories)))
    palette = sns.color_palette("tab20", len(unique_cats))
    cat_color_map = dict(zip(unique_cats, palette))

    for i, key in enumerate(keys):
        logger.info(f"Projecting {titles[i]}...")
        data_matrix = np.array(results[key])
        
        # Calculate Metrics
        if len(data_matrix) > 10000:
            indices = np.random.choice(len(data_matrix), 10000, replace=False)
            sil_score = silhouette_score(data_matrix[indices], categories[indices], metric='cosine')
        else:
            sil_score = silhouette_score(data_matrix, categories, metric='cosine')
            
        db_score = davies_bouldin_score(data_matrix, categories)
        
        metrics_results.append({
            'step': titles[i],
            'silhouette': sil_score,
            'davies_bouldin': db_score
        })
        
        title_with_metrics = f"{titles[i]}\nSil: {sil_score:.3f} | DB: {db_score:.3f}"
        
        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=100, min_dist=0.3)
            embedding = reducer.fit_transform(data_matrix)
        else:
            reducer = TSNE(n_components=2, random_state=42)
            embedding = reducer.fit_transform(data_matrix)
            
        # Plot
        ax = axes[i]
        
        # Plot background
        ax.scatter(embedding[:, 0], embedding[:, 1], c='lightgrey', s=5, alpha=0.3)
        
        # Plot top categories
        for cat in top_cats:
            idx = categories == cat
            ax.scatter(embedding[idx, 0], embedding[idx, 1], 
                      label=cat if i==1 else "", 
                      c=[cat_color_map[cat]], s=10, alpha=0.7)
            
        ax.set_title(title_with_metrics)
        ax.axis('off')

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(os.path.join(args.output_dir, 'gnn_separation_metrics.csv'), index=False)
    print("\n--- Separation Metrics ---")
    print(metrics_df)

    # Create a single legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cat_color_map[c], label=c, markersize=10) for c in top_cats]
    fig.legend(handles=handles, loc='center right', title="Orphanet Disease Category")
    plt.subplots_adjust(right=0.85)
    
    save_path = os.path.join(args.output_dir, 'gnn_latent_evolution_orphanet.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
