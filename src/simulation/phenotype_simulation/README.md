# Phenotype Simulation Module

Generates synthetic patient phenotype data based on empirical distributions of real patient characteristics.

## Overview

This module provides tools to simulate realistic phenotype sets for patients, based on:

1. **Empirical distribution** of the number of phenotypes observed per patient
2. **Empirical distribution** of phenotype distances from gene-specific phenotypes in the HPO hierarchy

The simulation process follows a modular design with the following components:

- **Distribution strategies** (e.g., `EmpiricalDistribution`): Models for sampling phenotype counts and distances
- **Phenotype selectors** (e.g., `HPODistancePhenotypeSelector`): Logic for choosing specific phenotypes for a gene
- **Simulator** (e.g., `StandardPhenotypeSimulator`): Main component that orchestrates the simulation process

## Key Features

- **Realistic Simulation**: Generates phenotype sets that reflect observed clinical patterns
- **Flexible Architecture**: Strategy pattern allows easy swapping of distribution models and selection logic
- **HPO Integration**: Uses HPO hierarchy to navigate phenotype distances
- **Batch Processing**: Efficiently simulates multiple patients for multiple genes

## Quick Start

```python
from src.simulation.phenotype_simulation.factory import SimulationFactory
from src.core.types import Phenotype

# Create a simulator with default components
# (Needs CSV files in data/simulation with phenotype count and distance distributions)
simulator = SimulationFactory.create_complete_simulator()

# Generate a single patient for a gene
patient_phenotypes = simulator.generate_patient("AARS1")

# Generate multiple patients for multiple genes
gene_to_count = {"AARS1": 10, "BRCA1": 20, "PAH": 15}
gene_patients = simulator.generate_patients_with_tqdm(gene_to_count)
```

## Demo Script

A comprehensive demo script is included to demonstrate the simulation process:

```bash
# Run with default settings
python -m src.simulation.phenotype_simulation.demo

# Customize the simulation
python -m src.simulation.phenotype_simulation.demo \
  --genes AARS1 BRCA1 PAH \
  --patients-per-gene 100 \
  --update-hpo \
  --regenerate-data
```

The demo:
1. Sets up necessary directories
2. Updates HPO data if needed
3. Prepares example distribution data (or uses existing CSV files)
4. Runs the simulation for specified genes
5. Saves the results to a JSON file
6. Prints statistics about the generated data

## Input Data Format

Two CSV files with empirical distributions:

1. **phenotype_counts.csv**: Contains a `count` column with the number of phenotypes per patient
2. **phenotype_distances.csv**: Contains a `distance` column with distances from specific phenotypes

## Extending the Module

### Custom Distribution Strategy

Create a new class implementing the `DistributionStrategy` interface:

```python
from src.simulation.phenotype_simulation.interfaces import DistributionStrategy

class GaussianDistribution(DistributionStrategy):
    """Distribution strategy based on Gaussian distribution."""
    
    def __init__(self, count_mean=5, count_std=1, distance_decay=0.5):
        self.count_mean = count_mean
        self.count_std = count_std
        self.distance_decay = distance_decay
        self._fitted = True
    
    def fit(self, data):
        # Optional: fit parameters from data
        pass
        
    def sample_phenotype_count(self):
        # Sample from Gaussian distribution
        count = int(np.random.normal(self.count_mean, self.count_std))
        return max(1, count)  # Ensure at least 1 phenotype
    
    def sample_distances(self, count):
        # Sample distances with exponential decay
        probs = np.exp(-np.arange(10) * self.distance_decay)
        probs = probs / probs.sum()
        return np.random.choice(len(probs), size=count, p=probs).tolist()
```

### Custom Phenotype Selector

Create a new class implementing the `PhenotypeSelector` interface:

```python
from src.simulation.phenotype_simulation.interfaces import PhenotypeSelector

class RandomPhenotypeSelector(PhenotypeSelector):
    """Selects random phenotypes regardless of gene."""
    
    def __init__(self, hpo_provider):
        self.hpo_provider = hpo_provider
        
    def select_phenotypes(self, gene, distances):
        # Get all phenotype IDs from HPO
        all_phenotypes = list(self.hpo_provider.get_all_terms())
        
        # Randomly select phenotypes
        selected_ids = np.random.choice(all_phenotypes, size=len(distances))
        
        # Convert to Phenotype objects
        phenotypes = []
        for phenotype_id in selected_ids:
            metadata = self.hpo_provider.get_metadata(phenotype_id)
            phenotypes.append(Phenotype(
                id=phenotype_id,
                name=metadata.get('name', ''),
                description=metadata.get('definition', '')
            ))
            
        return phenotypes
```

## Architecture

The module follows SOLID principles and uses the Strategy pattern for flexible component selection:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Distribution   │──────▶│    Simulator    │◀──────│   Phenotype     │
│   Strategy      │       │                 │       │    Selector     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
       ▲                         │                         ▲
       │                         │                         │
       │                         ▼                         │
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Empirical     │       │   Simulated     │       │  HPO Distance   │
│  Distribution   │       │    Patients     │       │    Selector     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

The Factory class provides convenient methods to create and configure all these components.

## Requirements

- numpy
- pandas
- tqdm (for progress tracking)
- HPO data (from `src.ontology` module)
- Gene-phenotype data (from `src.simulation.gene_phenotype` module) 