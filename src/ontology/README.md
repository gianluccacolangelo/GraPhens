# Human Phenotype Ontology (HPO) Module

Provides classes for managing and accessing the Human Phenotype Ontology (HPO), a standardized vocabulary for phenotypic abnormalities.

## Overview

This module ensures the system uses up-to-date HPO data in an efficient graph format.

- **`HPOGraphProvider`**: Loads HPO data (JSON/OBO) into a NetworkX graph, providing efficient traversal (ancestors, descendants) and metadata access. Includes global caching to avoid redundant loads.
- **`HPOUpdater`**: Downloads and updates local HPO data files from official sources (GitHub/JAX), managing versions.

## Basic Usage

```python
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater

# Update HPO data (checks interval, downloads if needed)
updater = HPOUpdater()
updater.check_for_updates(force=False)
updater.update(format_type="json") # Download latest JSON if needed

# Get a cached HPO graph instance
hpo_graph = HPOGraphProvider.get_instance(data_dir="data/ontology")

# Use the graph
ancestors = hpo_graph.get_ancestors("HP:0001250") # Seizure
metadata = hpo_graph.get_metadata("HP:0001250")
print(f"Name: {metadata.get('name')}, Definition: {metadata.get('definition')}")
```

## Key Features

- **Efficient Graph Access**: NetworkX DAG for fast hierarchy traversal.
- **Provider Caching**: `HPOGraphProvider.get_instance()` ensures the ontology is loaded only once.
- **Automatic Updates**: `HPOUpdater` keeps local data current.
- **Format Support**: Handles JSON and OBO ontology files.
- **Metadata Extraction**: Provides access to term names, definitions, etc.

## Integration

The `HPOGraphProvider` is primarily used by the `Augmentation` and `Graph` modules to understand phenotype relationships.

## Performance

- **Global Caching**: Significantly reduces startup time and memory usage when multiple components need the HPO graph.
- **JSON Preferred**: Loading from JSON is generally faster than OBO.

## Technical Details

### Graph Structure

- **Node Attributes**:
  - `id`: HPO term ID (e.g., "HP:0000123")
  - `name`: Human-readable term name
  - `definition`: Term definition (if available)
  - Additional metadata from the source file

- **Edge Direction**:
  - Edges point from child to parent for is-a relationships
  - This means `successors(term)` gives parents/ancestors
  - And `predecessors(term)` gives children/descendants

### Performance Optimizations

The `HPOGraphProvider` implements several optimizations to improve performance:

#### Global Provider Caching

- **Singleton-Like Pattern**: The class implements a `get_instance` method that returns cached instances
- **Shared Graph Data**: Multiple components can use the same HPO graph instance, avoiding redundant loading
- **Memory Efficiency**: The ontology is loaded only once, regardless of how many components need it
- **Automatic Loading**: Cached providers are automatically loaded when retrieved

Example usage:
```python
# Instead of creating new instances:
provider1 = HPOGraphProvider(data_dir="data/ontology")  # Creates new instance
provider2 = HPOGraphProvider(data_dir="data/ontology")  # Creates another instance

# Use the cached pattern:
provider1 = HPOGraphProvider.get_instance(data_dir="data/ontology")  # Creates or gets cached instance
provider2 = HPOGraphProvider.get_instance(data_dir="data/ontology")  # Returns same instance
```

#### Factory Integration

The component factory has been updated to use the cached instances:

```python
# Inside ComponentFactory
@classmethod
def create_hpo_provider(cls, config: Dict[str, Any]) -> Any:
    from src.ontology.hpo_graph import HPOGraphProvider
    
    data_dir = config.get("data_dir", "data/ontology")
    
    # Use the cached instance method instead of creating a new one
    return HPOGraphProvider.get_instance(data_dir=data_dir)
```

#### Performance Impact

This optimization significantly improves performance when multiple components access the HPO graph:

- **Startup Time**: Reduced by avoiding multiple ontology loads
- **Memory Usage**: Decreased by sharing a single graph instance
- **Batch Processing**: Critical for efficient processing of multiple patients

### File Formats

- **JSON**: Faster parsing, more compact memory representation
  - Supports both legacy "terms" format and newer "graphs/nodes/edges" format
- **OBO**: More standard in bioinformatics, used as fallback
  - Supports parsing with Pronto library (if available) for better results
  - Falls back to custom parsing if Pronto not installed

### Update Sources

- **GitHub**: Uses the GitHub API to check for new releases from the official HPO repository
- **JAX**: Directly queries the JAX HPO website for data files

## Performance Considerations

- JSON format provides the best performance for large ontologies
- Loading happens only once per session (cached after first load)
- NetworkX provides efficient traversal for ancestor/descendant queries
- Local data access is much faster than API-based alternatives 