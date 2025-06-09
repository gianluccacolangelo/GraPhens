# Phenotype Embedding Module

Provides strategies for embedding phenotypes (HPO terms) into vector representations suitable for GNN node features.

## Overview

This module uses the Strategy Pattern (`EmbeddingStrategy` interface from Core) to allow interchangeable embedding approaches.

## Available Strategies

- **Model-Based (On-the-fly generation):**
    - `HuggingFaceEmbeddingStrategy`: Uses any Hugging Face Transformer model.
    - `SentenceTransformerEmbeddingStrategy`: Uses models optimized for semantic similarity (e.g., `all-MiniLM-L6-v2`).
    - `OpenAIEmbeddingStrategy`: Uses OpenAI API models.
    - `LLMEmbeddingStrategy`: General wrapper for custom LLM services.
    - `TFIDFEmbeddingStrategy`: Lightweight TF-IDF vectorization.
- **Lookup-Based (Pre-computed):**
    - `LookupEmbeddingStrategy`: Uses embeddings loaded from a Python dictionary (Pickle file).
    - `MemmapEmbeddingStrategy`: Uses memory-mapped files for efficient loading and access of large pre-computed embedding sets.

## Usage

All strategies integrate via the `EmbeddingContext`:

```python
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import SentenceTransformerEmbeddingStrategy, LookupEmbeddingStrategy, MemmapEmbeddingStrategy

# Example 1: Use a SentenceTransformer model
strategy1 = SentenceTransformerEmbeddingStrategy(model_name="all-MiniLM-L6-v2")
context1 = EmbeddingContext(strategy1)
embeddings1 = context1.embed_phenotypes(phenotypes)

# Example 2: Use pre-computed embeddings (Pickle)
# with open("data/embeddings/hpo_embeddings.pkl", "rb") as f: embedding_dict = pickle.load(f)
# strategy2 = LookupEmbeddingStrategy(embedding_dict)
# context2 = EmbeddingContext(strategy2)
# embeddings2 = context2.embed_phenotypes(phenotypes)

# Example 3: Use memory-mapped embeddings (more efficient for large sets)
# Assumes memmap files exist (e.g., created via vector_db scripts)
# strategy3 = MemmapEmbeddingStrategy.from_latest(data_dir="data/embeddings")
# context3 = EmbeddingContext(strategy3)
# embeddings3 = context3.embed_phenotypes(phenotypes)
```

## Vector Database & Similarity

- **Precomputation:** The `src.embedding.vector_db` submodule provides tools (including CLI scripts) to precompute embeddings for the entire HPO and save them (Pickle or memory-mapped format) for use with lookup strategies.
    ```bash
    # Example: Build DB using a specific SentenceTransformer model
    python -m src.embedding.vector_db.scripts.build_vector_db --model "gsarti/biobert-nli" --model-type "sentence-transformers"
    ```
- **Similarity Search:** Tools are also available to find semantically similar phenotypes based on their vector embeddings, useful for clinical suggestions or analysis.
    ```bash
    # Example: Find 5 phenotypes similar to 'Seizure'
    python -m src.embedding.vector_db.scripts.find_similar_phenotypes HP:0001250 --num-similar 5
    ```

## Performance

- **`MemmapEmbeddingStrategy`**: Offers significant memory savings (3-7%) and faster access (15-20%) compared to standard `LookupEmbeddingStrategy` by leveraging memory-mapped files and lazy loading.
- **Import Optimization**: Careful management of optional dependencies reduces import time.

## Integration

The chosen `EmbeddingStrategy` (via `EmbeddingContext`) provides the `node_features` required by the `GraphAssembler` in the `graph` module.

## Recommended Models

For phenotype embeddings, these models have shown good performance:

- **Best Overall**: `gsarti/biobert-nli` - High variance, good discriminative power
- **Biomedical Domain**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` - Strong biomedical knowledge
- **Fast & Efficient**: `all-MiniLM-L6-v2` - Good performance with lower computational requirements
- **Clinical Focus**: `emilyalsentzer/Bio_ClinicalBERT` - Trained on clinical notes, good for clinical terms

According to our evaluation (see `results/REPORT.md`), `gsarti/biobert-nli` provides the best balance of semantic relatedness and discriminative power, making it particularly well-suited for GNN node features.

## Installation

To install the dependencies for all embedding strategies:

```bash
pip install -r requirements-embedding.txt
```

## Examples

## Using Vector Databases

### Specifying Vector Database Paths

When using the CLI tools, make sure to include the `.pkl` extension when specifying vector database paths:

```bash
# ❌ Incorrect - missing .pkl extension
python -m src.embedding.vector_db.scripts.find_similar_phenotypes \
  --vector-db data/embeddings/hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO_20250317_155952 \
  HP:0010779

# ✅ Correct - includes .pkl extension
python -m src.embedding.vector_db.scripts.find_similar_phenotypes \
  --vector-db data/embeddings/hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO_20250317_155952.pkl \
  HP:0010779
```

The vector database files are saved with the following naming pattern:
```
hpo_embeddings_<model_name>_<timestamp>.pkl
```

For example:
- `hpo_embeddings_gsarti_biobert-nli_20240320_120000.pkl`
- `hpo_embeddings_pritamdeka_S-PubMedBert-MS-MARCO_20250317_155952.pkl`

If you don't specify a vector database path, the system will automatically use the most recently created one in your `data/embeddings` directory. 

## Integration with Graph Module

The embedding strategies serve a crucial role in the graph construction pipeline by providing node features for phenotypes. This integration happens through the `EmbeddingContext` class, which follows the Strategy Pattern to make embedding strategies interchangeable.

### How It Works

1. Choose an embedding strategy based on your requirements (pre-computed lookups, on-the-fly generation, etc.)
2. Create an `EmbeddingContext` with your chosen strategy
3. Use the context to generate node features for your phenotypes
4. Pass these features to the `StandardGraphAssembler` along with edge indices to create a complete graph

```python
from src.embedding.context import EmbeddingContext
from src.embedding.strategies import LookupEmbeddingStrategy
from src.graph.assembler import StandardGraphAssembler
from src.graph.adjacency import HPOAdjacencyListBuilder
from src.augmentation.hpo_augmentation import HPOAugmentationService

# Initialize your embedding strategy
strategy = LookupEmbeddingStrategy(embedding_dict)

# Create the embedding context
context = EmbeddingContext(strategy)

# Load initial phenotypes
initial_phenotypes = [...]

# Optionally augment phenotypes with related terms
augmentation_service = HPOAugmentationService(
    data_dir="data/ontology", 
    include_ancestors=True,
    include_descendants=False
)
phenotypes = augmentation_service.augment(initial_phenotypes)

# Generate node features
node_features = context.embed_phenotypes(phenotypes)

# Build edge structure using HPO relationships
edge_index = adjacency_builder.build(phenotypes)

# Assemble the final graph
assembler = StandardGraphAssembler()
graph = assembler.assemble(
    phenotypes=phenotypes,
    node_features=node_features,
    edge_index=edge_index
)
```

### Flexible Architecture Benefits

This modular approach offers several advantages:

1. **Easy Experimentation**: Swap embedding strategies without changing any other code
2. **Performance Options**: Use lookup-based strategies for speed or model-based ones for quality
3. **Resource Adaptation**: Choose lightweight models for resource-constrained environments
4. **Domain Specialization**: Select biomedical domain models for better representation of phenotypes
5. **Seamless Updates**: When HPO terms are updated via the `HPOUpdater`, you can rebuild embeddings and graphs easily
6. **Augmentation Control**: Choose which related phenotypes to include through the augmentation service

### Example Demo

The `demo_assembler.py` script in the graph module demonstrates this integration by:

1. Loading phenotypes from the HPO ontology
2. Augmenting them with related terms using HPOAugmentationService
3. Using multiple embedding strategies (lookup-based and TFIDF-based)
4. Building adjacency lists with the `HPOAdjacencyListBuilder`
5. Assembling complete graphs with each type of embedding
6. Visualizing the resulting graphs

You can run the demo with:

```bash
python -m src.graph.demo_assembler --ontology-dir data/ontology --embeddings-dir data/embeddings
```

Options include:
- `--update-ontology`: Update the HPO ontology before running
- `--format`: Choose between 'json' or 'obo' ontology formats
- `--vector-db`: Specify a particular vector database file to use
- `--augment`: Augment phenotypes with related terms (default: true)
- `--include-ancestors`: Include ancestor terms in augmentation (default: true)
- `--include-descendants`: Include descendant terms in augmentation (default: false) 