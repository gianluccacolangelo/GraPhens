#!/usr/bin/env python3

import os
import logging
import argparse
from typing import List
from src.core.types import Phenotype
from src.augmentation.hpo_augmentation import HPOAugmentationService, APIAugmentationService
from src.ontology.hpo_graph import HPOGraphProvider
from src.ontology.hpo_updater import HPOUpdater

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("augmentation_demo")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate HPO augmentation services')
    
    parser.add_argument(
        '--data-dir', 
        default='data/ontology',
        help='Directory to store HPO data files (default: data/ontology)'
    )
    
    parser.add_argument(
        '--service', 
        choices=['local', 'api', 'both'],
        default='local',
        help='Augmentation service to use (default: local)'
    )
    
    parser.add_argument(
        '--include-descendants',
        action='store_true',
        help='Include descendant terms'
    )
    
    parser.add_argument(
        '--update-first',
        action='store_true',
        help='Update HPO data before running demo'
    )
    
    parser.add_argument(
        '--show-definitions',
        action='store_true',
        help='Show term definitions in the output'
    )
    
    parser.add_argument(
        'phenotypes',
        nargs='+',
        help='HPO IDs to augment (accepts both HP:NNNNNNN and HP_NNNNNNN formats)'
    )
    
    return parser.parse_args()

def normalize_hpo_id(hpo_id: str) -> str:
    """
    Normalize HPO ID format for consistency.
    
    Args:
        hpo_id: HPO ID in any format
        
    Returns:
        Normalized HPO ID
    """
    # Handle HP_ vs HP: format
    if hpo_id.startswith("HP_"):
        return hpo_id.replace("HP_", "HP:", 1)
    elif hpo_id.startswith("HP:"):
        return hpo_id
    else:
        # Try to add HP: prefix if it's just a number
        if hpo_id.isdigit():
            return f"HP:{hpo_id}"
        # Return as is if it doesn't match expected formats
        return hpo_id

def create_phenotypes(hpo_ids: List[str]) -> List[Phenotype]:
    """Create Phenotype objects from HPO IDs."""
    return [Phenotype(id=normalize_hpo_id(hpo_id), name="", description=None) for hpo_id in hpo_ids]

def main():
    """Run the augmentation demo."""
    args = parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Update HPO data if requested
    if args.update_first:
        logger.info("Updating HPO data...")
        updater = HPOUpdater(data_dir=args.data_dir)
        updater.update(format_type="json")
    
    # Create phenotype objects from input IDs
    input_phenotypes = create_phenotypes(args.phenotypes)
    
    # If showing definitions, initialize a graph provider to get input term details
    if args.show_definitions:
        hpo_graph = HPOGraphProvider(data_dir=args.data_dir)
        hpo_graph.load()
        
        # Update the input phenotypes with names and definitions
        for phenotype in input_phenotypes:
            metadata = hpo_graph.get_metadata(phenotype.id)
            phenotype.name = metadata.get('name', '')
            phenotype.description = metadata.get('definition', None)
    
    # Display input phenotypes
    print("\nInput phenotypes:")
    for p in input_phenotypes:
        if args.show_definitions and p.description:
            print(f"  - {p.id}: {p.name}\n    Definition: {p.description}")
        else:
            print(f"  - {p.id}: {p.name}")
    
    # Initialize services based on user selection
    services = []
    
    if args.service in ['local', 'both']:
        local_service = HPOAugmentationService(
            data_dir=args.data_dir,
            include_ancestors=True,
            include_descendants=args.include_descendants
        )
        services.append(("Local Service", local_service))
    
    if args.service in ['api', 'both']:
        api_service = APIAugmentationService()
        services.append(("API Service", api_service))
    
    # Run augmentation with each service
    for name, service in services:
        logger.info(f"Using {name}...")
        
        try:
            print(f"\n{name}:")
            print(f"Input: {len(input_phenotypes)} phenotypes")
            
            augmented = service.augment(input_phenotypes)
            
            print(f"Output: {len(augmented)} phenotypes")
            
            if len(augmented) > len(input_phenotypes):
                print("Added phenotypes:")
                
                # Display added phenotypes
                added_ids = set(p.id for p in augmented) - set(p.id for p in input_phenotypes)
                
                # If using local service, we can get names for the added phenotypes
                if name == "Local Service" and hasattr(service, "hpo_graph"):
                    for hpo_id in sorted(added_ids):
                        metadata = service.hpo_graph.get_metadata(hpo_id)
                        phenotype_name = metadata.get('name', '')
                        phenotype_definition = metadata.get('definition', None)
                        
                        if args.show_definitions and phenotype_definition:
                            print(f"  - {hpo_id}: {phenotype_name}\n    Definition: {phenotype_definition}")
                        else:
                            print(f"  - {hpo_id}: {phenotype_name}")
                else:
                    for hpo_id in sorted(added_ids):
                        # Find the phenotype in augmented list
                        for p in augmented:
                            if p.id == hpo_id:
                                phenotype_name = p.name
                                phenotype_definition = p.description
                                break
                        else:
                            phenotype_name = ""
                            phenotype_definition = None
                        
                        if args.show_definitions and phenotype_definition:
                            print(f"  - {hpo_id}: {phenotype_name}\n    Definition: {phenotype_definition}")
                        else:
                            print(f"  - {hpo_id}: {phenotype_name}")
            else:
                print("No additional phenotypes added")
                    
        except Exception as e:
            logger.error(f"Error using {name}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 