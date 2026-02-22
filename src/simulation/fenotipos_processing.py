import json
import os
from typing import List, Dict, Union, Any
from pathlib import Path
import re
from tqdm import tqdm
import gc
import torch

from src.core.types import Phenotype
from src.simulation.translation import load_embedding_resources, SimilarityMatch, cosine_similarity

"""
This script:

1. Takes a JSON file with gene phenotypes and processes it to convert natural language descriptions to HPO IDs

2. Handles both cases:
   - Already existing HPO IDs (preserves them with proper HP: prefix)
   - Natural language descriptions (converts to HPO IDs using the embedding similarity)

3. Includes error handling and marking:
   - `HP:0000123` - Successfully converted/existing HPO ID
   - `REVIEW:description` - No good match found (below similarity threshold)
   - `ERROR:description` - Error during processing

4. Provides detailed statistics about the processing results

To use it:

```bash
python -m src.simulation.fenotipos_processing
```

You can adjust:
- `model_name`: Choose which embedding model to use (biomedical models might work better)
- `min_similarity`: Set the minimum similarity threshold for accepting a match
- Input/output file paths

The output file will maintain the same structure as the input but with:
- Natural language descriptions converted to HPO IDs where possible
- Clear marking of entries that need review
- Proper HP: prefix on all HPO IDs

The script also provides progress bars and detailed statistics to help you monitor the conversion quality.

"""

def split_complex_description(description: str) -> List[str]:
    """
    Split a complex phenotype description into individual phenotypes.
    
    Args:
        description: A string containing one or more phenotype descriptions
        
    Returns:
        List of individual phenotype descriptions
    """
    # First, split by obvious sentence boundaries
    sentences = re.split(r'[.;,]', description)
    
    phenotypes = []
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Split by common connectors 
        parts = re.split(r'\s+(?:and|or|\+)\s+', sentence.strip())
        
        # Clean up each part and add to results
        for part in parts:
            cleaned = part.strip()
            # Remove common introductory phrases
            cleaned = re.sub(r'^(?:she|he|the patient|patient) (?:had|has|showed|presents with|developed)\s+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^(?:there was|there were)\s+', '', cleaned, flags=re.IGNORECASE)
            
            if cleaned:
                phenotypes.append(cleaned)
    
    return phenotypes

def process_phenotypes_file(
    input_file: str,
    output_file: str,
    model_name: str = "all-MiniLM-L6-v2",
    min_similarity: float = 0.5,
    batch_size: int = 100  # Process genes in batches
) -> None:
    """
    Process a phenotypes JSON file, converting natural language descriptions to HPO IDs.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        model_name: Name of the embedding model to use
        min_similarity: Minimum similarity score to accept a match
        batch_size: Number of genes to process in each batch
    """
    # Load input file
    print(f"Loading phenotypes from {input_file}")
    with open(input_file, 'r') as f:
        phenotypes_data = json.load(f)
    
    # Load embedding resources once (model, embeddings, HPO provider)
    print(f"Loading embedding resources using model: {model_name}")
    embedding_resources = load_embedding_resources(model_name)
    
    # Process each gene
    processed_data = {}
    total_genes = len(phenotypes_data)
    
    # Get list of genes for batch processing
    gene_list = list(phenotypes_data.keys())
    
    print(f"Processing {total_genes} genes in batches of {batch_size}...")
    for i in range(0, len(gene_list), batch_size):
        batch_genes = gene_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(gene_list) + batch_size - 1)//batch_size}...")
        
        for gene in tqdm(batch_genes):
            phenotype_lists = phenotypes_data[gene]
            processed_lists = []
            
            # Process each list of phenotypes for the gene
            for phenotype_list in phenotype_lists:
                processed_phenotypes = []
                
                # Process each phenotype
                for phenotype in phenotype_list:
                    # Clean up the phenotype string
                    phenotype = phenotype.strip()
                    
                    # Check if it's already an HPO ID
                    if is_hpo_id(phenotype):
                        # Add HP: prefix if missing
                        processed_phenotypes.append(format_hpo_id(phenotype))
                        continue
                    
                    # Split complex descriptions into individual phenotypes
                    individual_phenotypes = split_complex_description(phenotype)
                    
                    # If no split was possible or only one phenotype was found
                    if not individual_phenotypes:
                        individual_phenotypes = [phenotype]
                    
                    # Process each individual phenotype
                    for individual_phenotype in individual_phenotypes:
                        try:
                            matches = find_phenotype_matches_with_resources(
                                description=individual_phenotype,
                                resources=embedding_resources,
                                top_n=1
                            )
                            
                            if matches and matches[0].similarity >= min_similarity:
                                processed_phenotypes.append(matches[0].id)
                            else:
#                                if matches:
#                                    if matches[0].similarity > .8:
#                                        print(f"\nWarning: No good match found for '{individual_phenotype}' in gene {gene}")
#                                        print(f"Best match: {matches[0].id} ({matches[0].name}) with similarity {matches[0].similarity:.3f}")
                                processed_phenotypes.append(f"REVIEW:{individual_phenotype}")
                        except Exception as e:
                            print(f"\nError processing '{individual_phenotype}' in gene {gene}: {str(e)}")
                            processed_phenotypes.append(f"ERROR:{individual_phenotype}")
                
                processed_lists.append(processed_phenotypes)
            
            processed_data[gene] = processed_lists
        
        # Free memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Save intermediate results after each batch
        print(f"\nSaving intermediate results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
    
    # Save final processed data
    print(f"\nSaving final processed phenotypes to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Print statistics
    print_processing_stats(phenotypes_data, processed_data)

def find_phenotype_matches_with_resources(description: str, resources: Dict, top_n: int = 1):
    """Process a single phenotype using pre-loaded resources instead of loading them each time"""
    strategy = resources['strategy']
    embeddings_dict = resources['embeddings_dict']
    hpo_provider = resources['hpo_provider']
    
    # Create embedding for the description
    query_phenotype = Phenotype(id="query", name=description)
    query_embedding = strategy.embed_batch([query_phenotype])[0]
    
    # Compute cosine similarity with all pre-computed embeddings
    similarities = {}
    for hpo_id, embedding in embeddings_dict.items():
        similarity = cosine_similarity(query_embedding, embedding)
        similarities[hpo_id] = similarity
    
    # Get top N most similar phenotypes
    top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Format results
    results = []
    for hpo_id, similarity in top_matches:
        metadata = hpo_provider.get_metadata(hpo_id)
        match = SimilarityMatch(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            similarity=float(similarity)
        )
        results.append(match)
    
    return results

def is_hpo_id(text: str) -> bool:
    """Check if text is an HPO ID (with or without HP: prefix)."""
    # Remove HP: prefix if present
    text = text.replace("HP:", "").replace("hp:", "")
    # Check if it's a 7-digit number
    return bool(re.match(r'^\d{7}$', text))

def format_hpo_id(hpo_id: str) -> str:
    """Format HPO ID with HP: prefix."""
    # Remove any existing prefix and leading/trailing whitespace
    hpo_id = hpo_id.replace("HP:", "").replace("hp:", "").strip()
    return f"HP:{hpo_id}"

def print_processing_stats(original_data: Dict, processed_data: Dict) -> None:
    """Print statistics about the processing results."""
    total_phenotypes = 0
    converted_count = 0
    review_count = 0
    error_count = 0
    
    for gene, phenotype_lists in processed_data.items():
        for phenotype_list in phenotype_lists:
            for phenotype in phenotype_list:
                total_phenotypes += 1
                if phenotype.startswith("HP:"):
                    converted_count += 1
                elif phenotype.startswith("REVIEW:"):
                    review_count += 1
                elif phenotype.startswith("ERROR:"):
                    error_count += 1
    
    print("\nProcessing Statistics:")
    print(f"Total phenotypes processed: {total_phenotypes}")
    print(f"Successfully converted to HPO IDs: {converted_count} ({converted_count/total_phenotypes*100:.1f}%)")
    print(f"Needs review: {review_count} ({review_count/total_phenotypes*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/total_phenotypes*100:.1f}%)")

if __name__ == "__main__":
    # Example usage
    input_file = "src/simulation/genes_y_fenotipos.json"
    output_file = "src/simulation/updated_fenotipos_processed_all-MiniLM-L6-v2_fast.json"
    
    process_phenotypes_file(
        input_file=input_file,
        output_file=output_file,
        model_name="all-MiniLM-L6-v2",
        min_similarity=0.8 # Adjust this threshold based on results
    )
