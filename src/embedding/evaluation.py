"""
Evaluation module for biomedical embedding models.

This module provides tools to evaluate different embedding models on phenotype descriptions.
It samples phenotypes from the HPO ontology, generates embeddings, and analyzes
the distribution of pairwise similarities to identify models that provide the most variance.
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import pickle

from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
from src.embedding.strategies import (
    HuggingFaceEmbeddingStrategy,
    SentenceTransformerEmbeddingStrategy,
    OpenAIEmbeddingStrategy,
    LookupEmbeddingStrategy
)
from src.embedding.context import EmbeddingContext

# Top biomedical models based on MTEB performance
BIOMEDICAL_MODELS = {
    "sentence-transformers": [
        "pritamdeka/S-PubMedBert-MS-MARCO",
        "gsarti/biobert-nli",
        #"infgrad/jasper_en_vision_language_v1",
        #"dunzhang/stella_en_400M_v5",
        #"Salesforce/SFR-Embedding-2_R"
    ],
    "huggingface": [
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dmis-lab/biobert-base-cased-v1.2",
        "emilyalsentzer/Bio_ClinicalBERT",
        #"Alibaba-NLP/gte-Qwen2-7B-instruct"
    ]
}


class EmbeddingEvaluator:
    """Evaluates different embedding models on phenotype descriptions."""
    
    def __init__(self, 
                 data_dir: str = "data/embeddings",
                 ontology_dir: str = "data/ontology",
                 sample_size: int = 1000,
                 sample_pool_size: int = 9000,
                 random_seed: int = 42):
        """
        Initialize the embedding evaluator.
        
        Args:
            data_dir: Directory to store embedding data
            ontology_dir: Directory where HPO ontology data is stored
            sample_size: Number of phenotypes to sample
            sample_pool_size: Size of the pool to sample from
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.ontology_dir = ontology_dir
        self.sample_size = sample_size
        self.sample_pool_size = sample_pool_size
        self.random_seed = random_seed
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize HPO graph provider
        self.hpo_provider = HPOGraphProvider(data_dir=ontology_dir)
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Will store sampled phenotypes
        self.phenotypes: List[Phenotype] = []
        
        # Will store embeddings for each model
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Will store similarity matrices
        self.similarities: Dict[str, np.ndarray] = {}
        
        # Will store variance information
        self.variance_stats: Dict[str, Dict[str, float]] = {}
    
    def load_hpo_data(self) -> bool:
        """
        Load HPO ontology data.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        return self.hpo_provider.load()
    
    def sample_phenotypes(self) -> List[Phenotype]:
        """
        Randomly sample phenotypes from the HPO ontology.
        
        Returns:
            List of sampled phenotype objects
        """
        # Make sure HPO data is loaded
        if not self.hpo_provider.load():
            raise ValueError("Failed to load HPO data")
        
        # Get all term IDs
        all_terms = list(self.hpo_provider.terms.keys())
        
        # Limit to the first N terms
        pool_terms = all_terms[:min(self.sample_pool_size, len(all_terms))]
        
        # Randomly sample terms
        sample_size = min(self.sample_size, len(pool_terms))
        sampled_ids = random.sample(pool_terms, sample_size)
        
        # Create phenotype objects
        self.phenotypes = []
        for term_id in sampled_ids:
            metadata = self.hpo_provider.get_metadata(term_id)
            
            # Extract description - it might be in different fields depending on the format
            description = None
            if "definition" in metadata:
                description = metadata["definition"]
            elif "meta" in metadata and "definition" in metadata["meta"]:
                if isinstance(metadata["meta"]["definition"], dict):
                    description = metadata["meta"]["definition"].get("val", "")
                else:
                    description = str(metadata["meta"]["definition"])
            
            # Skip terms without descriptions
            if not description:
                continue
                
            # Clean up definition - sometimes it contains quotes and metadata
            if isinstance(description, str):
                # Remove quotes and everything after [
                if '"' in description:
                    description = description.split('"')[1] if len(description.split('"')) > 1 else description
                if '[' in description:
                    description = description.split('[')[0].strip()
            
            phenotype = Phenotype(
                id=term_id,
                name=metadata.get("name", ""),
                description=description
            )
            self.phenotypes.append(phenotype)
        
        print(f"Sampled {len(self.phenotypes)} phenotypes with descriptions")
        return self.phenotypes

    def get_embedding_strategy(self, model_type: str, model_name: str) -> Any:
        """
        Get the appropriate embedding strategy for the model.
        
        Args:
            model_type: Type of model ('sentence-transformers' or 'huggingface')
            model_name: Name of the model
            
        Returns:
            Embedding strategy instance
        """
        if model_type == "sentence-transformers":
            return SentenceTransformerEmbeddingStrategy(model_name=model_name)
        elif model_type == "huggingface":
            return HuggingFaceEmbeddingStrategy(model_name_or_path=model_name)
        elif model_type == "openai":
            return OpenAIEmbeddingStrategy(model=model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def generate_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all models.
        
        Returns:
            Dictionary mapping model names to embedding matrices
        """
        # Make sure we have phenotypes
        if not self.phenotypes:
            self.sample_phenotypes()
        
        # Check if cached embeddings exist and load them
        cache_file = os.path.join(self.data_dir, f"cached_embeddings_{self.sample_size}_{self.random_seed}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded cached embeddings from {cache_file}")
                
                # If we already have all models, return
                if all(model_name in self.embeddings for model_type in BIOMEDICAL_MODELS 
                      for model_name in BIOMEDICAL_MODELS[model_type]):
                    return self.embeddings
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")
                self.embeddings = {}
        
        # Generate embeddings for each model
        for model_type, model_names in BIOMEDICAL_MODELS.items():
            for model_name in model_names:
                # Skip if we already have embeddings for this model
                if model_name in self.embeddings:
                    print(f"Using cached embeddings for {model_name}")
                    continue
                
                print(f"Generating embeddings with {model_type} / {model_name}")
                
                try:
                    # Get the embedding strategy
                    strategy = self.get_embedding_strategy(model_type, model_name)
                    
                    # Create embedding context
                    embedding_context = EmbeddingContext(strategy)
                    
                    # Generate embeddings
                    start_time = datetime.now()
                    
                    # Use a progress bar
                    embeddings = embedding_context.embed_phenotypes(self.phenotypes)
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    print(f"Generated {embeddings.shape[0]} embeddings with {model_name} in {duration:.2f} seconds")
                    
                    # Store embeddings
                    self.embeddings[model_name] = embeddings
                    
                    # Cache intermediate results to avoid losing work
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.embeddings, f)
                    
                except Exception as e:
                    print(f"Error generating embeddings for {model_name}: {e}")
        
        return self.embeddings
    
    def calculate_similarities(self) -> Dict[str, np.ndarray]:
        """
        Calculate pairwise similarities for all models.
        
        Returns:
            Dictionary mapping model names to similarity matrices
        """
        # Make sure we have embeddings
        if not self.embeddings:
            self.generate_embeddings()
        
        # Calculate similarities for each model
        for model_name, embeddings in self.embeddings.items():
            print(f"Calculating similarities for {model_name}")
            similarity_matrix = cosine_similarity(embeddings)
            self.similarities[model_name] = similarity_matrix
            
            # Calculate statistics
            # For pairwise similarities, we only care about the upper triangle
            upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            self.variance_stats[model_name] = {
                "mean": np.mean(upper_tri),
                "std": np.std(upper_tri),
                "min": np.min(upper_tri),
                "max": np.max(upper_tri),
                "range": np.max(upper_tri) - np.min(upper_tri),
                "variance": np.var(upper_tri)
            }
            
            print(f"  Mean similarity: {self.variance_stats[model_name]['mean']:.4f}")
            print(f"  Std dev: {self.variance_stats[model_name]['std']:.4f}")
            print(f"  Range: {self.variance_stats[model_name]['range']:.4f}")
            print(f"  Variance: {self.variance_stats[model_name]['variance']:.4f}")
        
        return self.similarities
    
    def generate_histograms(self, output_dir: str = "results") -> List[str]:
        """
        Generate histograms of pairwise similarities.
        
        Args:
            output_dir: Directory to save the histograms
            
        Returns:
            List of paths to generated histogram files
        """
        # Make sure we have similarities
        if not self.similarities:
            self.calculate_similarities()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create histogram for each model
        histogram_files = []
        
        # Sort models by variance (highest first)
        sorted_models = sorted(
            self.variance_stats.keys(), 
            key=lambda x: self.variance_stats[x]["variance"], 
            reverse=True
        )
        
        # Create individual histograms
        for model_name in sorted_models:
            # Get upper triangle of similarity matrix (excluding diagonal)
            similarity_matrix = self.similarities[model_name]
            upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            plt.figure(figsize=(10, 6))
            plt.hist(upper_tri, bins=50, alpha=0.75, edgecolor='black')
            plt.title(f"Pairwise Similarity Distribution - {model_name}\nVariance: {self.variance_stats[model_name]['variance']:.6f}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # Add mean and std dev lines
            mean_val = self.variance_stats[model_name]["mean"]
            std_val = self.variance_stats[model_name]["std"]
            plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
            plt.axvline(mean_val + std_val, color='g', linestyle='dashed', linewidth=1, label=f'Mean + Std: {mean_val + std_val:.4f}')
            plt.axvline(mean_val - std_val, color='g', linestyle='dashed', linewidth=1, label=f'Mean - Std: {mean_val - std_val:.4f}')
            
            plt.legend()
            
            # Save histogram
            file_path = os.path.join(output_dir, f"similarity_histogram_{model_name.replace('/', '_')}.png")
            plt.savefig(file_path)
            plt.close()
            
            histogram_files.append(file_path)
            print(f"Generated histogram for {model_name}")
        
        # Create combined histogram (top 3 models by variance)
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Plot top 3 or fewer models
        for i, model_name in enumerate(sorted_models[:3]):
            similarity_matrix = self.similarities[model_name]
            upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            plt.hist(upper_tri, bins=50, alpha=0.5, edgecolor='black', 
                     label=f"{model_name} (Var: {self.variance_stats[model_name]['variance']:.6f})",
                     color=colors[i % len(colors)])
        
        plt.title("Comparison of Pairwise Similarity Distributions (Top Models by Variance)")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save combined histogram
        combined_path = os.path.join(output_dir, "combined_similarity_histogram.png")
        plt.savefig(combined_path)
        plt.close()
        
        histogram_files.append(combined_path)
        print(f"Generated combined histogram for top models")
        
        # Create box plot for all models
        plt.figure(figsize=(12, 8))
        
        # Prepare data for box plot
        box_data = []
        labels = []
        
        for model_name in sorted_models:
            similarity_matrix = self.similarities[model_name]
            upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            # Sample from upper_tri if it's too large to make the plot more efficient
            if len(upper_tri) > 10000:
                np.random.seed(self.random_seed)
                upper_tri = np.random.choice(upper_tri, 10000, replace=False)
                
            box_data.append(upper_tri)
            
            # Shorten model name for display
            display_name = model_name.split('/')[-1] if '/' in model_name else model_name
            var_val = self.variance_stats[model_name]["variance"]
            labels.append(f"{display_name}\n(Var: {var_val:.6f})")
        
        plt.boxplot(box_data, labels=labels, vert=True, patch_artist=True)
        plt.title("Distribution of Similarity Scores Across Models")
        plt.ylabel("Cosine Similarity")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Save box plot
        box_path = os.path.join(output_dir, "similarity_boxplot.png")
        plt.savefig(box_path, bbox_inches='tight')
        plt.close()
        
        histogram_files.append(box_path)
        print(f"Generated boxplot for all models")
        
        return histogram_files
    
    def run_evaluation(self, output_dir: str = "results") -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Run the complete evaluation workflow.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Tuple of variance statistics and list of histogram file paths
        """
        print(f"Starting evaluation with {self.sample_size} phenotypes...")
        
        # Load HPO data
        print("Loading HPO data...")
        self.load_hpo_data()
        
        # Sample phenotypes
        print(f"Sampling {self.sample_size} phenotypes...")
        self.sample_phenotypes()
        
        # Generate embeddings
        print("Generating embeddings...")
        self.generate_embeddings()
        
        # Calculate similarities
        print("Calculating similarities...")
        self.calculate_similarities()
        
        # Generate histograms
        print("Generating histograms...")
        histogram_files = self.generate_histograms(output_dir)
        
        # Print results summary
        print("\nResults Summary:")
        print("================")
        sorted_models = sorted(
            self.variance_stats.keys(), 
            key=lambda x: self.variance_stats[x]["variance"], 
            reverse=True
        )
        
        for model_name in sorted_models:
            stats = self.variance_stats[model_name]
            print(f"{model_name}:")
            print(f"  Variance: {stats['variance']:.6f}")
            print(f"  Mean: {stats['mean']:.4f} (Std: {stats['std']:.4f})")
            print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
            print()
        
        print(f"Generated {len(histogram_files)} visualization files in {output_dir}")
        
        return self.variance_stats, histogram_files
    
    def save_embeddings_to_vector_db(self, db_path: str = None) -> str:
        """
        Save embeddings to a simple vector database (pickle file).
        
        Args:
            db_path: Path to save the vector database
            
        Returns:
            Path to the vector database
        """
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"vector_db_{self.sample_size}_{self.random_seed}.pkl")
        
        # Make sure we have embeddings
        if not self.embeddings:
            self.generate_embeddings()
        
        # Prepare vector database
        vector_db = {
            "embeddings": self.embeddings,
            "phenotypes": [
                {"id": p.id, "name": p.name, "description": p.description}
                for p in self.phenotypes
            ],
            "metadata": {
                "sample_size": self.sample_size,
                "random_seed": self.random_seed,
                "timestamp": datetime.now().isoformat(),
                "models": list(self.embeddings.keys())
            }
        }
        
        # Save the vector database
        with open(db_path, 'wb') as f:
            pickle.dump(vector_db, f)
        
        print(f"Saved vector database to {db_path}")
        return db_path

def run_evaluator(
    sample_size: int = 1000,
    sample_pool_size: int = 9000,
    random_seed: int = 42,
    data_dir: str = "data/embeddings",
    ontology_dir: str = "data/ontology",
    output_dir: str = "results/embedding_evaluation",
    save_vector_db: bool = True
) -> None:
    """
    Run the embedding evaluator with the specified parameters.
    
    Args:
        sample_size: Number of phenotypes to sample
        sample_pool_size: Size of the pool to sample from
        random_seed: Random seed for reproducibility
        data_dir: Directory to store embedding data
        ontology_dir: Directory where HPO ontology data is stored
        output_dir: Directory to save results
        save_vector_db: Whether to save embeddings to a vector database
    """
    # Create evaluator
    evaluator = EmbeddingEvaluator(
        data_dir=data_dir,
        ontology_dir=ontology_dir,
        sample_size=sample_size,
        sample_pool_size=sample_pool_size,
        random_seed=random_seed
    )
    
    # Run evaluation
    variance_stats, histogram_files = evaluator.run_evaluation(output_dir)
    
    # Save vector database if requested
    if save_vector_db:
        evaluator.save_embeddings_to_vector_db()
    
if __name__ == "__main__":
    # This allows running the module directly for testing
    run_evaluator(sample_size=1000, random_seed=42) 