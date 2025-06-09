# GraPhens Configuration Guide

GraPhens provides a flexible configuration system using YAML files, allowing you to customize all aspects of the system without modifying code. This guide explains the available configuration options and how to use them.

## Configuration Structure

A GraPhens configuration file is organized into the following sections:

- `data`: Paths to data directories
- `augmentation`: Settings for phenotype augmentation
- `embedding`: Settings for embedding generation
- `graph`: Settings for graph construction
- `visualization`: Settings for graph visualization
- `logging`: Settings for logging
- `performance`: Settings for performance optimization

## Basic Usage

To use a configuration file:

```python
from src.graphens import GraPhens

# Initialize with a configuration file
graphens = GraPhens(config_path="config/default.yaml")

# Or load a configuration after initialization
graphens = GraPhens().load_config_from_yaml("config/custom_config.yaml")

# Save the current configuration
graphens.save_config_to_yaml("config/my_config.yaml")
```

## Configuration Options

### Data Directories

```yaml
data:
  ontology_dir: "data/ontology"
  embeddings_dir: "data/embeddings"
  visualizations_dir: "visualizations"
```

| Option | Description | Default |
|--------|-------------|---------|
| `ontology_dir` | Directory for HPO ontology files | `"data/ontology"` |
| `embeddings_dir` | Directory for embedding files | `"data/embeddings"` |
| `visualizations_dir` | Directory for visualization output | `"visualizations"` |

### Ontology Settings

```yaml
ontology:
  update_on_start: false
  format: "json"
  use_github: true
  check_interval: 604800  # 7 days in seconds
```

| Option | Description | Default |
|--------|-------------|---------|
| `update_on_start` | Check for HPO updates on startup | `false` |
| `format` | Preferred HPO format (`"json"` or `"obo"`) | `"json"` |
| `use_github` | Use GitHub as the HPO source | `true` |
| `check_interval` | Time between update checks (seconds) | `604800` (7 days) |

### Augmentation Settings

```yaml
augmentation:
  type: "local"
  include_ancestors: true
  include_descendants: false
  api_base_url: "https://example.com/api/augment"  # Only used if type is "api"
```

| Option | Description | Default |
|--------|-------------|---------|
| `type` | Augmentation strategy (`"local"` or `"api"`) | `"local"` |
| `include_ancestors` | Include ancestor terms in augmentation | `true` |
| `include_descendants` | Include descendant terms in augmentation | `false` |
| `api_base_url` | Base URL for API augmentation | (required for `"api"` type) |

### Embedding Settings

```yaml
embedding:
  type: "sentence_transformer"  # One of: memmap, lookup, sentence_transformer, huggingface, tfidf, openai
  
  # Settings for memmap embeddings
  memmap:
    dir: "data/embeddings"
  
  # Settings for lookup embeddings
  lookup:
    embedding_file: "data/embeddings/embeddings.pkl"
  
  # Settings for sentence_transformer embeddings
  sentence_transformer:
    model_name: "all-MiniLM-L6-v2"
    batch_size: 32
  
  # Settings for huggingface embeddings
  huggingface:
    model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
    max_length: 128
    use_gpu: false
    batch_size: 32
  
  # Settings for TF-IDF embeddings
  tfidf:
    max_features: 768
    ngram_range: [1, 2]
  
  # Settings for OpenAI embeddings
  openai:
    model: "text-embedding-3-small"
    api_key: ""  # Defaults to OPENAI_API_KEY environment variable if empty
```

| Option | Description | Default |
|--------|-------------|---------|
| `type` | Embedding strategy to use | `"sentence_transformer"` |

Specific settings depend on the chosen embedding type:

#### Memory-mapped Embeddings (`"memmap"`)

| Option | Description | Default |
|--------|-------------|---------|
| `dir` | Directory containing memory-mapped embedding files | `"data/embeddings"` |

#### Lookup Embeddings (`"lookup"`)

| Option | Description | Default |
|--------|-------------|---------|
| `embedding_file` | Path to the embeddings file | (required) |

#### Sentence Transformer Embeddings (`"sentence_transformer"`)

| Option | Description | Default |
|--------|-------------|---------|
| `model_name` | Name of the Sentence Transformers model | `"all-MiniLM-L6-v2"` |
| `batch_size` | Batch size for processing | `32` |

#### Hugging Face Embeddings (`"huggingface"`)

| Option | Description | Default |
|--------|-------------|---------|
| `model_name_or_path` | Model name or path | (required) |
| `max_length` | Maximum sequence length | `128` |
| `use_gpu` | Whether to use GPU for inference | `false` |
| `batch_size` | Batch size for processing | `32` |

#### TF-IDF Embeddings (`"tfidf"`)

| Option | Description | Default |
|--------|-------------|---------|
| `max_features` | Maximum number of features | `768` |
| `ngram_range` | Range of n-grams to include | `[1, 2]` |

#### OpenAI Embeddings (`"openai"`)

| Option | Description | Default |
|--------|-------------|---------|
| `model` | OpenAI embedding model | `"text-embedding-3-small"` |
| `api_key` | OpenAI API key | (uses environment variable) |

### Graph Settings

```yaml
graph:
  adjacency:
    include_reverse_edges: true
  
  assembler:
    validate: true
  
  global_context:
    type: "root"  # One of: null, "root", "subgraph"
    include_root: false  # Only for "subgraph" type
```

| Option | Description | Default |
|--------|-------------|---------|
| `adjacency.include_reverse_edges` | Include reverse edges in the graph | `true` |
| `assembler.validate` | Validate the assembled graph | `true` |
| `global_context.type` | Type of global context to add | `null` |
| `global_context.include_root` | Include root node in subgraph context | `false` |

### Visualization Settings

```yaml
visualization:
  enabled: false
  type: "graphviz"  # One of: "graphviz", "networkx"
  format: "png"     # One of: "png", "svg", "pdf"
  output_dir: "visualizations"
  limit_nodes: 100  # Maximum number of nodes to visualize
```

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable visualization | `false` |
| `type` | Visualization backend | `"graphviz"` |
| `format` | Output file format | `"png"` |
| `output_dir` | Directory for output files | `"visualizations"` |
| `limit_nodes` | Maximum number of nodes to visualize | `100` |

### Logging Settings

```yaml
logging:
  level: "INFO"  # One of: DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  to_file: false
  log_file: "graphens.log"
```

| Option | Description | Default |
|--------|-------------|---------|
| `level` | Logging level | `"INFO"` |
| `format` | Log message format | (standard format) |
| `to_file` | Whether to log to a file | `false` |
| `log_file` | Path to the log file | `"graphens.log"` |

### Performance Settings

```yaml
performance:
  cache_hpo_provider: true
  cache_orchestrator: true
  enable_batch_processing: true
```

| Option | Description | Default |
|--------|-------------|---------|
| `cache_hpo_provider` | Cache the HPO provider | `true` |
| `cache_orchestrator` | Cache the orchestrator | `true` |
| `enable_batch_processing` | Enable efficient batch processing | `true` |

## Complete Example

Here's a complete configuration file with all options:

```yaml
data:
  ontology_dir: "data/ontology"
  embeddings_dir: "data/embeddings"
  visualizations_dir: "visualizations"

ontology:
  update_on_start: false
  format: "json"
  use_github: true
  check_interval: 604800

augmentation:
  type: "local"
  include_ancestors: true
  include_descendants: false
  api_base_url: "https://example.com/api/augment"

embedding:
  type: "sentence_transformer"
  
  memmap:
    dir: "data/embeddings"
  
  lookup:
    embedding_file: "data/embeddings/embeddings.pkl"
  
  sentence_transformer:
    model_name: "all-MiniLM-L6-v2"
    batch_size: 32
  
  huggingface:
    model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
    max_length: 128
    use_gpu: false
    batch_size: 32
  
  tfidf:
    max_features: 768
    ngram_range: [1, 2]
  
  openai:
    model: "text-embedding-3-small"
    api_key: ""

graph:
  adjacency:
    include_reverse_edges: true
  
  assembler:
    validate: true
  
  global_context:
    type: null
    include_root: false

visualization:
  enabled: false
  type: "graphviz"
  format: "png"
  output_dir: "visualizations"
  limit_nodes: 100

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  to_file: false
  log_file: "graphens.log"

performance:
  cache_hpo_provider: true
  cache_orchestrator: true
  enable_batch_processing: true
```

## Environment Variables

Some configuration options can be set using environment variables:

- `OPENAI_API_KEY`: Used for OpenAI embeddings if not specified in the configuration
- `GRAPHENS_CONFIG_PATH`: Default path to the configuration file
- `GRAPHENS_DATA_DIR`: Default path to the data directory

## Programmatic Configuration

Instead of using YAML files, you can configure GraPhens programmatically:

```python
graphens = GraPhens()

# Configure embedding strategy
graphens.with_memmap_embeddings()
# or
graphens.with_pretrained_embeddings("path/to/embeddings.pkl")
# or
graphens.with_sentence_transformer_embeddings("all-MiniLM-L6-v2")

# Configure augmentation
graphens.with_local_augmentation(include_ancestors=True, include_descendants=False)
# or
graphens.with_api_augmentation(api_base_url="https://example.com/augment")

# Configure graph structure
graphens.with_adjacency_settings(include_reverse_edges=True)
graphens.with_global_context("root")

# Configure visualization
graphens.with_visualization(output_dir="visualizations", format="png")
```

Any configuration set programmatically will override the corresponding settings from the YAML file. 