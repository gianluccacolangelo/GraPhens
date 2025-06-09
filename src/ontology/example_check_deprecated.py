#!/usr/bin/env python3
import logging
from src.ontology.check_deprecated import CheckDeprecated

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example usage of the CheckDeprecated class."""
    
    # Initialize the CheckDeprecated class
    checker = CheckDeprecated()
    
    # Example from the prompt: HP:0040263 (obsolete Jaw ankylosis) -> HP:0012478
    deprecated_id = "HP:0040263"
    
    # Check if term is deprecated
    is_deprecated = checker.is_deprecated(deprecated_id)
    logger.info(f"Is {deprecated_id} deprecated? {is_deprecated}")
    
    # Get replacement if deprecated
    if is_deprecated:
        replacement_id = checker.get_replacement(deprecated_id)
        if replacement_id:
            logger.info(f"Replacement for {deprecated_id} is {replacement_id}")
            
            # Show the label of the replacement
            if replacement_id in checker.terms:
                replacement_label = checker.terms[replacement_id].get("label", "")
                logger.info(f"Replacement term label: {replacement_label}")
        else:
            logger.info(f"No replacement found for {deprecated_id}")
    
    # Alternate usage with check_and_replace
    logger.info("\nUsing check_and_replace method:")
    is_dep, replacement = checker.check_and_replace(deprecated_id)
    logger.info(f"Term {deprecated_id} - Deprecated: {is_dep}, Replacement: {replacement}")
    
    # Check a non-deprecated term
    non_deprecated_id = "HP:0000001"  # Typically "All" or root term
    is_dep, replacement = checker.check_and_replace(non_deprecated_id)
    logger.info(f"Term {non_deprecated_id} - Deprecated: {is_dep}, Replacement: {replacement}")
    
    # Testing different ID formats
    logger.info("\nTesting different ID formats:")
    # With underscore
    id_with_underscore = "HP_0040263"
    is_dep, replacement = checker.check_and_replace(id_with_underscore)
    logger.info(f"Term {id_with_underscore} - Deprecated: {is_dep}, Replacement: {replacement}")
    
    # Full URI
    id_uri = "http://purl.obolibrary.org/obo/HP_0040263"
    is_dep, replacement = checker.check_and_replace(id_uri)
    logger.info(f"Term {id_uri} - Deprecated: {is_dep}, Replacement: {replacement}")
    
if __name__ == "__main__":
    main() 