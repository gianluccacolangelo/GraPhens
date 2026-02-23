from typing import List, Set
from src.core.interfaces import AugmentationService
from src.core.types import Phenotype
from src.ontology.hpo_graph import HPOGraphProvider
import requests
import logging

"""
This module provides two implementations of the AugmentationService interface
for enhancing phenotype sets using the Human Phenotype Ontology (HPO):

1. HPOAugmentationService: Uses a local HPO directed acyclic graph (DAG) for traversal
2. APIAugmentationService: Uses the JAX HPO Ontology REST API for fetching ancestors

The augmentation process enriches the input data by adding related terms
(typically ancestors or parent terms) from the HPO hierarchy. This allows
the system to capture more comprehensive relationships between phenotypes
in the resulting graph, improving disease matching and analysis.

Both implementations follow the same interface, allowing for flexible
deployment depending on whether a local HPO database is available or
if the application needs to use the remote API.

The APIAugmentationService includes error handling and logging to manage
API failures gracefully, ensuring that even if some terms cannot be augmented,
the original phenotypes are preserved.
"""

class APIAugmentationService(AugmentationService):
    """
    Augments phenotypes using the HPO Ontology REST API.
    
    This service calls the JAX HPO Ontology API to retrieve ancestor terms
    for each input phenotype, adding these terms to the phenotype set.
    
    Attributes:
        api_base_url: Base URL for the HPO API (default: https://ontology.jax.org/api/hp)
        logger: Logger instance for error and debug information
    """
    
    def __init__(self, api_base_url="https://ontology.jax.org/api/hp"):
        """
        Initialize with the API base URL.
        
        Args:
            api_base_url: Base URL for the HPO API
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
    
    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """
        Augment the list of observed phenotypes by fetching their ancestors from the API.
        
        For each phenotype in the input list, retrieves its ancestors from the HPO
        ontology and adds them to the result. The original phenotypes are preserved
        even if API errors occur.
        
        Args:
            observed: List of phenotypes to augment
            
        Returns:
            List of phenotypes including the original set plus ancestors
        """
        observed_ids = {p.id for p in observed}
        augmented_ids = observed_ids.copy()
        
        # Collect ancestor IDs for each phenotype
        for phenotype in observed:
            try:
                ancestors = self._get_ancestors(phenotype.id)
                augmented_ids.update(ancestors)
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Error fetching ancestors for {phenotype.id}: {str(e)}")
        
        # Create new phenotypes for added IDs
        additional_phenotypes = []
        for hpo_id in augmented_ids - observed_ids:
            try:
                phenotype = self._create_phenotype(hpo_id)
                additional_phenotypes.append(phenotype)
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Error creating phenotype for {hpo_id}: {str(e)}")
        
        return observed + additional_phenotypes
    
    def _get_ancestors(self, hpo_id: str) -> Set[str]:
        """
        Fetch ancestor terms for an HPO ID using the API.
        
        Args:
            hpo_id: HPO term ID (e.g., "HP:0032120")
            
        Returns:
            Set of HPO IDs representing ancestor terms
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        endpoint = f"{self.api_base_url}/terms/{hpo_id}/ancestors"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            # Extract HPO IDs from the response
            ancestors = {term['id'] for term in response.json()}
            return ancestors
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.warning(f"HPO ID {hpo_id} not found, skipping ancestors")
                return set()
            raise
    
    def _create_phenotype(self, hpo_id: str) -> Phenotype:
        """
        Create a phenotype object by fetching metadata from the API.
        
        Args:
            hpo_id: HPO term ID (e.g., "HP:0032120")
            
        Returns:
            Phenotype object populated with data from the API
            
        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        endpoint = f"{self.api_base_url}/terms/{hpo_id}"
        response = requests.get(endpoint)
        response.raise_for_status()
        
        term_data = response.json()
        return Phenotype(
            id=hpo_id,
            name=term_data.get('name', ''),
            description=term_data.get('definition', None),
            metadata=term_data
        )

class HPOAugmentationService(AugmentationService):
    """
    Augments phenotypes using a local HPO DAG traversal.
    
    This service uses the HPOGraphProvider to access the HPO ontology
    and add related terms based on configuration options.
    
    Attributes:
        hpo_graph: Provider for accessing the HPO directed acyclic graph
        include_ancestors: Whether to include ancestor terms
        include_descendants: Whether to include descendant terms
    """
    
    def __init__(self, data_dir: str = "data/ontology", include_ancestors=True, include_descendants=False):
        """
        Initialize with HPO graph provider and traversal options.
        
        Args:
            data_dir: Directory where HPO data files are stored
            include_ancestors: Whether to include ancestor terms (default: True)
            include_descendants: Whether to include descendant terms (default: False)
        """
        # Use get_instance to leverage caching
        self.hpo_graph = HPOGraphProvider.get_instance(data_dir=data_dir)
        self.include_ancestors = include_ancestors
        self.include_descendants = include_descendants
        self.logger = logging.getLogger(__name__)
        self._graph_ready = self.hpo_graph.last_loaded is not None or self.hpo_graph.load()
    
    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """
        Augment the list of observed phenotypes using the HPO DAG.
        
        Args:
            observed: List of phenotypes to augment
            
        Returns:
            List of phenotypes including the original set plus related terms
        """
        if not self._graph_ready:
            self.logger.error("Failed to load HPO graph, returning original phenotypes")
            return observed
            
        observed_ids = {p.id for p in observed}
        augmented_ids = observed_ids.copy()
        
        # Collect IDs to add based on the configuration
        for phenotype in observed:
            if self.include_ancestors:
                ancestors = self.hpo_graph.get_ancestors(phenotype.id)
                augmented_ids.update(ancestors)
            
            if self.include_descendants:
                descendants = self.hpo_graph.get_descendants(phenotype.id)
                augmented_ids.update(descendants)
        
        # Create new phenotypes for added IDs
        additional_phenotypes = []
        for hpo_id in augmented_ids - observed_ids:
            try:
                phenotype = self._create_phenotype(hpo_id)
                additional_phenotypes.append(phenotype)
            except Exception as e:
                self.logger.warning(f"Error creating phenotype for {hpo_id}: {str(e)}")
        
        return observed + additional_phenotypes
    
    def _create_phenotype(self, hpo_id: str) -> Phenotype:
        """
        Create a phenotype object from an HPO ID.
        
        Args:
            hpo_id: HPO term ID
            
        Returns:
            Phenotype object populated with data from the local graph
        """
        metadata = self.hpo_graph.get_metadata(hpo_id)
        return Phenotype(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            metadata=metadata
        )

class SiblingsAugmentationService(AugmentationService):
    """
    Augments phenotypes by adding their siblings from the HPO graph.

    This service finds the siblings of each phenotype by first identifying
    the direct parents and then finding all children of those parents.

    Attributes:
        hpo_graph: Provider for accessing the HPO directed acyclic graph.
    """
    def __init__(self, data_dir: str = "data/ontology"):
        """
        Initialize with HPO graph provider.

        Args:
            data_dir: Directory where HPO data files are stored.
        """
        self.hpo_graph = HPOGraphProvider.get_instance(data_dir=data_dir)
        self.logger = logging.getLogger(__name__)
        self._graph_ready = self.hpo_graph.last_loaded is not None or self.hpo_graph.load()

    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """
        Augment the list of observed phenotypes by adding their siblings.

        Args:
            observed: List of phenotypes to augment.

        Returns:
            List of phenotypes including the original set plus their siblings.
        """
        if not self._graph_ready:
            self.logger.error("Failed to load HPO graph, returning original phenotypes")
            return observed

        observed_ids = {p.id for p in observed}
        all_siblings = set()

        # Collect all siblings for all observed phenotypes
        for phenotype in observed:
            parents = self.hpo_graph.get_direct_parents(phenotype.id)
            for parent_id in parents:
                children_of_parent = self.hpo_graph.get_direct_children(parent_id)
                all_siblings.update(children_of_parent)
        
        # Determine which phenotypes to add
        new_phenotype_ids = all_siblings - observed_ids
        
        # Create new phenotype objects
        additional_phenotypes = []
        for hpo_id in new_phenotype_ids:
            try:
                phenotype = self._create_phenotype(hpo_id)
                additional_phenotypes.append(phenotype)
            except Exception as e:
                self.logger.warning(f"Error creating phenotype for {hpo_id}: {str(e)}")

        return observed + additional_phenotypes

    def _create_phenotype(self, hpo_id: str) -> Phenotype:
        """
        Create a phenotype object from an HPO ID.

        Args:
            hpo_id: HPO term ID.

        Returns:
            Phenotype object populated with data from the local graph.
        """
        metadata = self.hpo_graph.get_metadata(hpo_id)
        return Phenotype(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            metadata=metadata
        )

class NHopAugmentationService(AugmentationService):
    """
    Augments phenotypes by finding all nodes within an N-hop distance in the HPO graph.

    This service performs a bidirectional traversal (parents and children) from each
    initial phenotype up to a specified number of hops (N).

    Attributes:
        hpo_graph: Provider for accessing the HPO directed acyclic graph.
        n_hops: The maximum number of hops to traverse from the initial nodes.
    """
    def __init__(self, n_hops: int, data_dir: str = "data/ontology"):
        """
        Initialize with the number of hops and HPO graph provider.

        Args:
            n_hops: The number of hops to traverse.
            data_dir: Directory where HPO data files are stored.
        """
        if n_hops < 1:
            raise ValueError("n_hops must be a positive integer.")
            
        self.hpo_graph = HPOGraphProvider.get_instance(data_dir=data_dir)
        self.n_hops = n_hops
        self.logger = logging.getLogger(__name__)

    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """
        Augment the list of observed phenotypes by finding all nodes within N-hops.

        Args:
            observed: List of phenotypes to augment.

        Returns:
            List of phenotypes including the original set and all nodes within N-hops.
        """
        if not self.hpo_graph.load():
            self.logger.error("Failed to load HPO graph, returning original phenotypes")
            return observed

        observed_ids = {p.id for p in observed}
        
        # --- N-hop traversal using Breadth-First Search (BFS) ---
        # Queue stores tuples of (hpo_id, current_distance)
        queue = [(p.id, 0) for p in observed]
        
        # Visited set keeps track of nodes we've already processed to avoid cycles/redundancy
        visited = set(observed_ids)
        
        # This will store all the new phenotype IDs we find
        n_hop_ids = set()

        while queue:
            current_id, distance = queue.pop(0)

            # If we've reached the N-hop limit for this path, we stop exploring further
            if distance >= self.n_hops:
                continue

            # Get neighbors (both parents and children for a bidirectional search)
            # Parents are successors because edges are child -> parent
            parents = self.hpo_graph.graph.successors(current_id)
            # Children are predecessors
            children = self.hpo_graph.graph.predecessors(current_id)
            
            neighbors = set(parents) | set(children)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    n_hop_ids.add(neighbor_id)
                    queue.append((neighbor_id, distance + 1))
        
        # Create new phenotype objects for the IDs found
        additional_phenotypes = []
        # We only add phenotypes that were not in the original observed list
        for hpo_id in n_hop_ids - observed_ids:
            try:
                phenotype = self._create_phenotype(hpo_id)
                additional_phenotypes.append(phenotype)
            except Exception as e:
                self.logger.warning(f"Error creating phenotype for {hpo_id}: {str(e)}")

        return observed + additional_phenotypes

    def _create_phenotype(self, hpo_id: str) -> Phenotype:
        """
        Create a phenotype object from an HPO ID.

        Args:
            hpo_id: HPO term ID.

        Returns:
            Phenotype object populated with data from the local graph.
        """
        metadata = self.hpo_graph.get_metadata(hpo_id)
        return Phenotype(
            id=hpo_id,
            name=metadata.get('name', ''),
            description=metadata.get('definition', None),
            metadata=metadata
        )

class CompositeAugmentationService(AugmentationService):
    """
    Applies a sequence of augmentation services.

    This service allows for chaining multiple augmentation strategies together.
    Each service is applied sequentially to the output of the previous one.
    """
    def __init__(self, services: List[AugmentationService]):
        """
        Initialize with a list of augmentation services.

        Args:
            services: A list of augmentation service instances to apply in order.
        """
        self.services = services
        self.logger = logging.getLogger(__name__)

    def augment(self, observed: List[Phenotype]) -> List[Phenotype]:
        """
        Augment phenotypes by applying each service in the chain.

        Args:
            observed: The initial list of phenotypes.

        Returns:
            The final list of phenotypes after all augmentations have been applied.
        """
        augmented_phenotypes = observed
        for i, service in enumerate(self.services):
            service_name = service.__class__.__name__
            self.logger.debug(f"Applying augmentation chain step {i+1}/{len(self.services)}: {service_name}")
            augmented_phenotypes = service.augment(augmented_phenotypes)
        
        return augmented_phenotypes
