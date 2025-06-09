"""
Analysis of gene-phenotype relationships.

This script analyzes the distribution of phenotypes in the dataset and their relationships
in the HPO graph hierarchy. It computes and visualizes:
1. Distribution of observed phenotypes per case
2. Proportion of specific vs. non-specific phenotypes
3. Distance of observed phenotypes from the closest specific phenotype
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

# Import GraPhens utilities
from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
from src.augmentation.hpo_augmentation import HPOAugmentationService
from src.simulation.gene_phenotype import GenePhenotypeFacade
from src.visualization.graphviz import GraphvizVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhenotypeAnalyzer:
    """
    Analyzes phenotype distributions and relationships in the HPO graph.
    """
    
    def __init__(self, 
                 fenotipos_file: str,
                 gene_phenotype_dir: str = "data/",
                 ontology_dir: str = "data/ontology",
                 output_dir: str = "results"):
        """
        Initialize the phenotype analyzer.
        
        Args:
            fenotipos_file: Path to the processed phenotypes JSON file
            gene_phenotype_dir: Directory containing gene-phenotype database files
            ontology_dir: Directory containing HPO ontology files
            output_dir: Directory where to save outputs
        """
        self.fenotipos_file = fenotipos_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load case data
        logger.info(f"Loading phenotype cases from {fenotipos_file}")
        with open(fenotipos_file, 'r') as f:
            self.case_data = json.load(f)
            
        # Create GraPhens components
        self.facade = GenePhenotypeFacade(data_dir=gene_phenotype_dir)
        self.hpo_provider = HPOGraphProvider.get_instance(data_dir=ontology_dir)
        self.augmentation_service = HPOAugmentationService(
            data_dir=ontology_dir,
            include_ancestors=True,
            include_descendants=False
        )
        
        # Initialize results containers
        self.phenotypes_per_case = []
        self.specific_proportions = []
        self.phenotype_distances = []
        
    def analyze_phenotype_distributions(self):
        """
        Analyze the distribution of phenotypes across cases.
        """
        logger.info("Analyzing phenotype distributions...")
        
        # Add counters for tracking analysis results
        total_genes = 0
        total_cases = 0
        cases_with_phenotypes = 0
        cases_with_specific_phenotypes = 0
        genes_without_specific_phenotypes = 0
        
        # Iterate through genes and their cases
        for gene, cases in self.case_data.items():
            total_genes += 1
            # Get specific phenotypes for this gene
            specific_phenotypes = set()
            try:
                gene_phenotypes = self.facade.get_phenotypes_for_gene(gene)
                specific_phenotypes = {p.id for p in gene_phenotypes}
                if not specific_phenotypes:
                    logger.warning(f"Gene {gene} has no specific phenotypes in database")
                    genes_without_specific_phenotypes += 1
                if gene == "A2ML1":
                    logger.info(f"Specific phenotypes for {gene}: {specific_phenotypes}")
                    logger.info(f"Gene phenotypes: {gene_phenotypes}")
            except Exception as e:
                logger.warning(f"Could not get specific phenotypes for gene {gene}: {str(e)}")
                genes_without_specific_phenotypes += 1
                continue
                
            # Count valid cases for this gene
            gene_cases = 0
            gene_cases_with_phenotypes = 0
            gene_cases_with_specific = 0
            
            # Process each case for this gene
            for case in cases:
                total_cases += 1
                gene_cases += 1
                
                # Skip cases with no phenotypes or review comments
                if not case or (len(case) == 1 and case[0].startswith("REVIEW:")):
                    continue
                    
                # Process observed phenotypes
                observed_phenotypes = set(case)
                observed_count = len(observed_phenotypes)
                
                # Skip cases with no valid phenotypes
                if observed_count == 0:
                    continue
                
                # Count this as a case with phenotypes
                cases_with_phenotypes += 1
                gene_cases_with_phenotypes += 1
                
                # Calculate specific phenotypes observed
                specific_observed = observed_phenotypes.intersection(specific_phenotypes)
                specific_count = len(specific_observed)
                
                # If there are specific phenotypes, count it
                if specific_count > 0:
                    cases_with_specific_phenotypes += 1
                    gene_cases_with_specific += 1
                
                # Calculate proportion of specific phenotypes
                specific_proportion = specific_count / observed_count if observed_count > 0 else 0
                
                # Save to results
                self.phenotypes_per_case.append(observed_count)
                self.specific_proportions.append(specific_proportion)
                
                # Calculate distance distribution for this case
                self._analyze_phenotype_distances(gene, observed_phenotypes, specific_phenotypes)
            
            # Log gene-level stats
            if gene_cases > 0:
                logger.debug(f"Gene {gene}: {gene_cases_with_phenotypes}/{gene_cases} cases with phenotypes, " +
                           f"{gene_cases_with_specific}/{gene_cases_with_phenotypes} with specific phenotypes")
        
        # Log summary statistics
        logger.info(f"Analyzed {total_genes} genes with {total_cases} total cases")
        logger.info(f"Cases with phenotypes: {cases_with_phenotypes}/{total_cases} ({cases_with_phenotypes/total_cases*100:.1f}%)")
        logger.info(f"Cases with specific phenotypes: {cases_with_specific_phenotypes}/{cases_with_phenotypes} ({cases_with_specific_phenotypes/cases_with_phenotypes*100:.1f}%)")
        logger.info(f"Genes without specific phenotypes: {genes_without_specific_phenotypes}/{total_genes} ({genes_without_specific_phenotypes/total_genes*100:.1f}%)")
        logger.info(f"Average phenotypes per case: {np.mean(self.phenotypes_per_case):.2f}")
        logger.info(f"Average proportion of specific phenotypes: {np.mean(self.specific_proportions):.2f}")
        if hasattr(self, 'disconnected_phenotypes'):
            logger.info(f"Number of disconnected phenotypes: {self.disconnected_phenotypes}")
    def _analyze_phenotype_distances(self, gene: str, observed: Set[str], specific: Set[str]):
        """
        Analyze distances between observed phenotypes and specific phenotypes in the HPO graph.
        
        For each observed phenotype, this method:
        1. Builds a subgraph with specific phenotypes from gene-phenotype data
        2. For each observed phenotype:
           - If it's a leaf node in the HPO graph, distance = 0
           - Otherwise, traverses DOWN the hierarchy to find closest leaf
           - Counts steps to closest leaf node going only downward
           - If no path exists (phenotype is disconnected from graph), 
             the phenotype is counted as disconnected
           
        Args:
            gene: Gene symbol
            observed: Set of observed phenotype IDs
            specific: Set of specific phenotype IDs for this gene
        """
        # DEBUG: Log gene and phenotype counts
        logger.debug(f"Gene {gene}: observed={len(observed)}, specific={len(specific)}")
        logger.debug(f"Observed phenotypes: {observed}")
        logger.debug(f"Specific phenotypes: {specific}")
        
        if not specific:
            logger.warning(f"Gene {gene} has no specific phenotypes, skipping.")
            return
            
        # Count how many observed phenotypes are also specific
        overlap = observed.intersection(specific)
        logger.debug(f"Gene {gene}: {len(overlap)}/{len(observed)} observed phenotypes are specific")
            
        # Initialize dictionary to track disconnected phenotypes if not already done
        if not hasattr(self, 'disconnected_phenotypes_dict'):
            self.disconnected_phenotypes_dict = {}
            self.disconnected_phenotypes = 0
            
        # Build the complete HPO subgraph for this gene
        # First, create a graph with all nodes and edges from the HPO
        G = nx.DiGraph()
        
        # Add all HPO nodes and their relationships
        # Focus on the specific phenotypes and their ancestors/descendants
        nodes_to_add = set(specific)
        
        # Add ancestors and descendants of specific phenotypes to ensure connectedness
        for hpo_id in specific:
            if hpo_id in self.hpo_provider.graph:
                # Add ancestors
                for ancestor in nx.ancestors(self.hpo_provider.graph, hpo_id):
                    nodes_to_add.add(ancestor)
                # Add descendants
                for descendant in nx.descendants(self.hpo_provider.graph, hpo_id):
                    nodes_to_add.add(descendant)
                # Add the node itself
                nodes_to_add.add(hpo_id)
            else:
                logger.warning(f"Specific phenotype {hpo_id} not found in HPO graph")
        
        # DEBUG: Log subgraph node count
        logger.debug(f"Gene {gene}: Subgraph has {len(nodes_to_add)} nodes")
        
        # Create the subgraph with these nodes
        for hpo_id in nodes_to_add:
            G.add_node(hpo_id)
            
            # Add edges to parents (going up in the hierarchy)
            for parent in self.hpo_provider.graph.successors(hpo_id):
                if parent in nodes_to_add:
                    # Edge direction: child -> parent (HPO graph direction)
                    G.add_edge(hpo_id, parent)
        
        # DEBUG: Count edges in subgraph
        logger.debug(f"Gene {gene}: Subgraph has {G.number_of_edges()} edges")
        
        # Add logging variables to track phenotype categorization
        specific_count = 0
        connected_count = 0
        disconnected_count = 0
        not_in_hpo_count = 0
        
        # Now process each observed phenotype
        for hpo_id in observed:
            phenotype_name = self.hpo_provider.get_metadata(hpo_id).get('name', 'Unknown')
            
            # Skip phenotypes not in HPO graph
            if hpo_id not in self.hpo_provider.graph:
                logger.warning(f"Observed phenotype {hpo_id} ({phenotype_name}) not in HPO graph")
                not_in_hpo_count += 1
                continue
            
            # If this is a specific phenotype, distance is 0
            if hpo_id in specific:
                logger.debug(f"Phenotype {hpo_id} ({phenotype_name}) is specific, distance=0")
                self.phenotype_distances.append(0)
                specific_count += 1
                continue
            
            # DEBUG: We know this phenotype is not specific
            logger.debug(f"Processing non-specific phenotype {hpo_id} ({phenotype_name})")
            
            # Check if the observed phenotype is in our subgraph
            # If not, add it and connect to its parents
            if hpo_id not in G:
                G.add_node(hpo_id)
                
                # Connect to parents in the subgraph
                connected = False
                for parent in self.hpo_provider.graph.successors(hpo_id):
                    if parent in G:
                        G.add_edge(hpo_id, parent)
                        connected = True
                        parent_name = self.hpo_provider.get_metadata(parent).get('name', 'Unknown')
                        logger.debug(f"Connected {hpo_id} to parent {parent} ({parent_name})")
                
                # If we couldn't connect it to any parent, it's disconnected
                if not connected:
                    self.disconnected_phenotypes += 1
                    disconnected_count += 1
                    # Track which phenotypes are disconnected
                    if gene not in self.disconnected_phenotypes_dict:
                        self.disconnected_phenotypes_dict[gene] = []
                    self.disconnected_phenotypes_dict[gene].append((hpo_id, phenotype_name))
                    logger.debug(f"Phenotype {hpo_id} ({phenotype_name}) is disconnected - no parents in subgraph")
                    continue
            
            # Now find the distance to the closest specific phenotype by traversing 
            # BOTH UP and DOWN the hierarchy
            # We need to check both ancestors and descendants
            
            # First find all descendants of this phenotype in the graph
            descendants = set()
            to_check = [hpo_id]
            visited = {hpo_id}
            
            while to_check:
                current = to_check.pop(0)
                # Get children of current node (predecessors in G)
                for child in G.predecessors(current):
                    if child not in visited:
                        visited.add(child)
                        to_check.append(child)
                        descendants.add(child)
            
            # Then find all ancestors of this phenotype in the graph
            ancestors = set()
            to_check = [hpo_id]
            visited = {hpo_id}
            
            while to_check:
                current = to_check.pop(0)
                # Get parents of current node (successors in G)
                for parent in G.successors(current):
                    if parent not in visited:
                        visited.add(parent)
                        to_check.append(parent)
                        ancestors.add(parent)
            
            # DEBUG: Log relatives count
            logger.debug(f"Phenotype {hpo_id} has {len(descendants)} descendants and {len(ancestors)} ancestors in subgraph")
            
            # Check if any descendants are specific phenotypes
            specific_descendants = descendants.intersection(specific)
            # Check if any ancestors are specific phenotypes
            specific_ancestors = ancestors.intersection(specific)
            
            # Combine both sets of specific relatives
            specific_relatives = specific_descendants.union(specific_ancestors)
            
            if specific_relatives:
                logger.debug(f"Found {len(specific_relatives)} specific relatives for {hpo_id} ({len(specific_descendants)} descendants, {len(specific_ancestors)} ancestors)")
                # We need to find the shortest path to any of these specific phenotypes
                min_distance = float('inf')
                closest_specific = None
                
                for spec_id in specific_relatives:
                    try:
                        # Try to find shortest path from hpo_id to spec_id or spec_id to hpo_id
                        # depending on whether it's an ancestor or descendant
                        if spec_id in specific_descendants:
                            # For descendants, we go from parent to child (reverse direction)
                            path_length = nx.shortest_path_length(G.reverse(), source=hpo_id, target=spec_id)
                        else:
                            # For ancestors, we go from child to parent (original direction)
                            path_length = nx.shortest_path_length(G, source=hpo_id, target=spec_id)
                            
                        if path_length < min_distance:
                            min_distance = path_length
                            closest_specific = spec_id
                    except nx.NetworkXNoPath:
                        logger.debug(f"No path between {hpo_id} and specific phenotype {spec_id}")
                        continue
            
                if min_distance != float('inf'):
                    self.phenotype_distances.append(min_distance)
                    connected_count += 1
                    closest_name = self.hpo_provider.get_metadata(closest_specific).get('name', 'Unknown')
                    logger.debug(f"Distance from {hpo_id} to closest specific {closest_specific} ({closest_name}) is {min_distance}")
                    
                    # Log more detailed information about this connected non-specific phenotype
                    logger.debug(f"CONNECTED: {hpo_id} ({phenotype_name}) -> {closest_specific} ({closest_name}), distance={min_distance}")
                    
                    # If this is the first connected non-specific phenotype for this gene, create a dictionary
                    if not hasattr(self, 'connected_phenotypes_dict'):
                        self.connected_phenotypes_dict = {}
                    
                    if gene not in self.connected_phenotypes_dict:
                        self.connected_phenotypes_dict[gene] = []
                    
                    # Store the information
                    self.connected_phenotypes_dict[gene].append({
                        'phenotype_id': hpo_id,
                        'phenotype_name': phenotype_name,
                        'connected_to': closest_specific,
                        'connected_name': closest_name,
                        'distance': min_distance,
                        'direction': 'descendant' if closest_specific in specific_descendants else 'ancestor'
                    })
                else:
                    # Shouldn't happen but handle just in case
                    self.disconnected_phenotypes += 1
                    disconnected_count += 1
                    if gene not in self.disconnected_phenotypes_dict:
                        self.disconnected_phenotypes_dict[gene] = []
                    self.disconnected_phenotypes_dict[gene].append((hpo_id, phenotype_name))
                    logger.warning(f"No path found to any specific phenotype for {hpo_id} despite having specific relatives")
            else:
                # No specific phenotypes among either ancestors or descendants
                self.disconnected_phenotypes += 1
                disconnected_count += 1
                if gene not in self.disconnected_phenotypes_dict:
                    self.disconnected_phenotypes_dict[gene] = []
                self.disconnected_phenotypes_dict[gene].append((hpo_id, phenotype_name))
                logger.debug(f"No specific phenotypes found among relatives of {hpo_id}")
        
        # Log summary for this gene
        logger.debug(f"Gene {gene} phenotype summary: {specific_count} specific, {connected_count} connected non-specific, " +
                    f"{disconnected_count} disconnected, {not_in_hpo_count} not in HPO")
        
    def plot_distributions(self):
        """
        Plot the phenotype distribution analyses.
        """
        logger.info("Plotting distributions...")
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Distribution of phenotypes per case
        axs[0].hist(self.phenotypes_per_case, bins=20, alpha=0.7)
        axs[0].set_title('Phenotypes per Case')
        axs[0].set_xlabel('Number of Phenotypes')
        axs[0].set_ylabel('Number of Cases')
        axs[0].grid(alpha=0.3)
        
        # Plot 2: Distribution of specific phenotype proportions
        axs[1].hist(self.specific_proportions, bins=20, alpha=0.7)
        axs[1].set_title('Proportion of Specific Phenotypes')
        axs[1].set_xlabel('Proportion')
        axs[1].set_ylabel('Number of Cases')
        axs[1].grid(alpha=0.3)
        
        # Plot 3: Distribution of phenotype distances
        # Filter out any very large distances that might be outliers
        filtered_distances = [d for d in self.phenotype_distances if d < 10]  
        distance_counter = Counter(filtered_distances)
        
        # Create bar plot for distances
        distances = sorted(distance_counter.keys())
        counts = [distance_counter[d] for d in distances]
        
        # Plot connected phenotypes in blue
        axs[2].bar(distances, counts, alpha=0.7, color='royalblue', label='Connected')
        
        # Add disconnected phenotypes as a separate bar in red
        if hasattr(self, 'disconnected_phenotypes') and self.disconnected_phenotypes > 0:
            axs[2].bar(-1, self.disconnected_phenotypes, alpha=0.7, color='crimson', label='Disconnected')
            # Adjust x-axis to include the disconnected bar
            axs[2].set_xticks([-1] + distances)
            axs[2].set_xticklabels(['Disc.'] + [str(d) for d in distances])
        else:
            axs[2].set_xticks(distances)
            
        axs[2].set_title('Distance to Closest Specific Phenotype')
        axs[2].set_xlabel('Distance (Nodes)')
        axs[2].set_ylabel('Count')
        axs[2].grid(alpha=0.3)
        axs[2].legend()
        # Add text with stats
        plt.figtext(0.5, 0.01, f'Total Cases: {len(self.phenotypes_per_case)}, ' +
                   f'Avg Phenotypes: {np.mean(self.phenotypes_per_case):.2f}, ' +
                   f'Avg Specific Proportion: {np.mean(self.specific_proportions):.2f}',
                   ha='center', fontsize=12)
        
        # Save the plot
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'phenotype_distributions.png')
        plt.savefig(output_path, dpi=300)
        logger.info(f"Plot saved to {output_path}")
        
        # Close the plot to free memory
        plt.close(fig)
        
    def save_results(self):
        """
        Save analysis results to CSV files.
        """
        logger.info("Saving analysis results...")
        
        # Save phenotypes per case
        df_counts = pd.DataFrame({'phenotypes_per_case': self.phenotypes_per_case})
        df_counts.to_csv(os.path.join(self.output_dir, 'phenotypes_per_case.csv'), index=False)
        
        # Save specific proportions
        df_proportions = pd.DataFrame({'specific_proportion': self.specific_proportions})
        df_proportions.to_csv(os.path.join(self.output_dir, 'specific_proportions.csv'), index=False)
        
        # Save distance distribution
        df_distances = pd.DataFrame({'distance': self.phenotype_distances})
        df_distances.to_csv(os.path.join(self.output_dir, 'phenotype_distances.csv'), index=False)
        
        # Save connected phenotypes report
        if hasattr(self, 'connected_phenotypes_dict'):
            with open(os.path.join(self.output_dir, 'connected_phenotypes.json'), 'w') as f:
                json.dump(self.connected_phenotypes_dict, f, indent=2)
            
            # Create a readable text report
            with open(os.path.join(self.output_dir, 'connected_phenotypes_report.txt'), 'w') as f:
                total_connected = sum(len(phenotypes) for phenotypes in self.connected_phenotypes_dict.values())
                f.write(f"Total connected non-specific phenotypes: {total_connected}\n\n")
                for gene, phenotypes in self.connected_phenotypes_dict.items():
                    f.write(f"Gene: {gene} ({len(phenotypes)} connected phenotypes)\n")
                    for p in phenotypes:
                        f.write(f"  - {p['phenotype_id']} ({p['phenotype_name']}) -> {p['connected_to']} ({p['connected_name']}), " +
                                f"distance={p['distance']}, direction={p['direction']}\n")
                    f.write("\n")
            
            logger.info(f"Saved connected phenotypes report with {total_connected} phenotypes")
        
        # Save disconnected phenotypes report
        if hasattr(self, 'disconnected_phenotypes_dict'):
            with open(os.path.join(self.output_dir, 'disconnected_phenotypes.json'), 'w') as f:
                json.dump(self.disconnected_phenotypes_dict, f, indent=2)
            
            # Create a readable text report
            with open(os.path.join(self.output_dir, 'disconnected_phenotypes_report.txt'), 'w') as f:
                f.write(f"Total disconnected phenotypes: {self.disconnected_phenotypes}\n\n")
                for gene, phenotypes in self.disconnected_phenotypes_dict.items():
                    f.write(f"Gene: {gene} ({len(phenotypes)} disconnected phenotypes)\n")
                    for hpo_id, name in phenotypes:
                        f.write(f"  - {hpo_id}: {name}\n")
                    f.write("\n")
            
        logger.info(f"Results saved to {self.output_dir}")
        
    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting phenotype analysis...")
        
        # Run analyses
        self.analyze_phenotype_distributions()
        
        # Create visualizations
        self.plot_distributions()
        
        # Save results
        self.save_results()
        
        logger.info("Analysis complete.")

    def visualize_gene_phenotypes(self, gene_symbol: str):
        """
        Visualize the phenotypes for a specific gene, showing the hierarchy.
        
        Args:
            gene_symbol: The gene symbol to visualize
        
        Returns:
            Path to the generated visualization file
        """
        logger.info(f"Visualizing phenotype hierarchy for gene {gene_symbol}...")
        
        # Get specific phenotypes for this gene
        try:
            specific_phenotype_objects = self.facade.get_phenotypes_for_gene(gene_symbol)
        except Exception as e:
            logger.error(f"Could not get specific phenotypes for gene {gene_symbol}: {str(e)}")
            return None
        
        # Get all cases for this gene from the fenotipos data
        if gene_symbol not in self.case_data:
            logger.error(f"Gene {gene_symbol} not found in case data")
            return None
        
        # Collect all observed phenotypes from cases
        observed_phenotype_ids = set()
        for case in self.case_data[gene_symbol]:
            # Skip review comments
            if len(case) == 1 and case[0].startswith("REVIEW:"):
                continue
            observed_phenotype_ids.update(case)
        
        # Create Phenotype objects for observed phenotypes
        observed_phenotype_objects = [
            Phenotype(
                id=hpo_id, 
                name=self.hpo_provider.get_metadata(hpo_id).get('name', 'Unknown')
            )
            for hpo_id in observed_phenotype_ids
        ]
        
        # Augment the specific phenotypes to get ancestors
        augmented_phenotypes = self.augmentation_service.augment(specific_phenotype_objects)
        
        # Initialize the visualizer
        visualizer = GraphvizVisualizer(
            output_dir=self.output_dir,
            hpo_provider=self.hpo_provider
        )
        
        # Create visualizations
        # 1. Specific phenotypes and their hierarchy
        specific_viz_path = visualizer.visualize_hierarchy(
            phenotypes=specific_phenotype_objects,
            title=f"{gene_symbol} - Specific Phenotypes"
        )
        
        # 2. Augmented phenotypes with specific ones highlighted
        augmented_viz_path = visualizer.visualize_augmentation_result(
            augmented_phenotypes=augmented_phenotypes,
            initial_phenotypes=specific_phenotype_objects,
            title=f"{gene_symbol} - Augmented Phenotypes"
        )
        
        # 3. Observed phenotypes from cases
        observed_viz_path = visualizer.visualize_hierarchy(
            phenotypes=observed_phenotype_objects,
            title=f"{gene_symbol} - Observed Phenotypes"
        )
        
        # 4. Combined visualization with both specific and observed
        combined_phenotypes = augmented_phenotypes + [
            p for p in observed_phenotype_objects 
            if p.id not in {ap.id for ap in augmented_phenotypes}
        ]
        
        combined_viz_path = visualizer.visualize_hierarchy(
            phenotypes=combined_phenotypes,
            initial_phenotypes=specific_phenotype_objects,
            title=f"{gene_symbol} - Combined Phenotypes"
        )
        
        logger.info(f"Created visualizations for gene {gene_symbol}")
        logger.info(f"Specific phenotypes: {specific_viz_path}")
        logger.info(f"Augmented phenotypes: {augmented_viz_path}")
        logger.info(f"Observed phenotypes: {observed_viz_path}")
        logger.info(f"Combined phenotypes: {combined_viz_path}")
        
        return {
            'specific': specific_viz_path,
            'augmented': augmented_viz_path,
            'observed': observed_viz_path,
            'combined': combined_viz_path
        }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze phenotype distributions")
    parser.add_argument("--fenotipos-file", default="src/simulation/fenotipos_cleaned_processed.json", 
                        help="Path to the processed phenotypes JSON file")
    parser.add_argument("--gene-phenotype-dir", default="data/", 
                        help="Directory containing gene-phenotype database files")
    parser.add_argument("--ontology-dir", default="data/ontology", 
                        help="Directory containing HPO ontology files")
    parser.add_argument("--output-dir", default="results", 
                        help="Directory where to save outputs")
    parser.add_argument("--visualize-gene", 
                        help="Visualize phenotypes for a specific gene (e.g., 'A2ML1')")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PhenotypeAnalyzer(
        fenotipos_file=args.fenotipos_file,
        gene_phenotype_dir=args.gene_phenotype_dir,
        ontology_dir=args.ontology_dir,
        output_dir=args.output_dir
    )
    
    if args.visualize_gene:
        # Only visualize the specific gene
        analyzer.visualize_gene_phenotypes(args.visualize_gene)
    else:
        # Run the full analysis
        analyzer.run_analysis()


if __name__ == "__main__":
    main() 