import os
import torch
import logging
import json
import lmdb
import pickle
import numpy as np
from pathlib import Path
from torch_geometric.data import Dataset, Data
from typing import List, Tuple, Optional, Dict, Any, Union
import io

# Added: Define or import the constant used in convert_to_lmdb
# Ensure this value matches the one in convert_to_lmdb.py
EXPECTED_SAMPLES_PER_BATCH = 5000 

logger = logging.getLogger(__name__)
# Configure basic logging if run standalone
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LMDBHPOGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for HPO graphs with LMDB backend.
    
    This dataset uses LMDB (Lightning Memory-Mapped Database) for fast random access
    to graph data. It provides significant performance improvements over loading
    individual files, especially for random batch access during training.
    """

    def __init__(self, 
                root_dir: str, 
                transform=None, 
                pre_transform=None,
                readonly: bool = True,
                map_size: int = 1099511627776):  # 1TB default map size
        """
        Initialize the LMDB-backed HPO dataset.

        Args:
            root_dir: The root directory of the dataset containing the LMDB files
            transform: PyTorch Geometric transform to apply to data upon access
            pre_transform: PyTorch Geometric transform (not typically used here)
            readonly: Whether to open LMDB in readonly mode (should be True for training)
            map_size: Size of the LMDB map (only relevant when creating the database)
        """
        logger.debug("LMDBHPOGraphDataset __init__ started.")

        # Pass root, transform, pre_transform to the base class
        super().__init__(root_dir, transform, pre_transform)
        # Note: The base class __init__ sets self.root, self.transform, self.pre_transform
        # We might be overwriting them below, which is fine, but be aware.
        self.root_dir = Path(root_dir).resolve() # Already set by super? Check base class if needed.
        logger.debug(f"Set root_dir: {self.root_dir}")
        self._db_path = self.root_dir / "hpo_lmdb"
        logger.debug(f"Set _db_path: {self._db_path}")
        self._metadata_path = self.root_dir / "metadata.json"
        self._metadata: Optional[Dict[str, Any]] = None
        self._db = None
        self._txn = None
        self._readonly = readonly
        self._map_size = map_size
        self._length = None
        self._gene_idx_map = None  # For efficient gene lookups

        logger.info(f"Initializing LMDBHPOGraphDataset with LMDB at: {self._db_path}")

        # Create database directory if it doesn't exist (needed for conversion)
        if not self._readonly and not self._db_path.exists():
            self._db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created LMDB directory: {self._db_path}")

        # Check if LMDB exists when in readonly mode
        if self._readonly and not (self._db_path / "data.mdb").exists():
            raise FileNotFoundError(
                f"LMDB database not found at {self._db_path}. "
                "Run the conversion script to create it from .pt files."
            )

        # Load metadata
        self._load_metadata()

        # Get dataset length
        self._get_length()

        # Store path and options needed for opening later
        self.worker_id = None # Initialize

        logger.info(f"Successfully initialized LMDB dataset with {len(self)} samples")

    def _open_db_if_needed(self):
        """Open the LMDB database connection if not already open for the current worker."""
        current_worker_id = torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0 # 0 for main process
        if self._db is None or self.worker_id != current_worker_id:
            logger.debug(f"Opening LMDB for worker {current_worker_id} (PID: {os.getpid()})")
            self.worker_id = current_worker_id
            try:
                # Close previous if exists from a different worker context (though maybe not needed)
                if self._db: self._db.close() 

                self._db = lmdb.open(
                    str(self._db_path),
                    readonly=self._readonly,
                    map_size=self._map_size, # map_size might not be needed if readonly=True
                    max_dbs=1,
                    lock=False, 
                    readahead=False, 
                    meminit=False  
                )
                # Ensure transaction is opened immediately after environment
                self._txn = self._db.begin(write=False) 
                logger.debug(f"LMDB Opened successfully for worker {current_worker_id} and transaction started.")
            except Exception as e:
                logger.error(f"Failed to open LMDB in worker {current_worker_id}: {e}")
                # Reset db and txn on failure to avoid using stale handles
                self._db = None
                self._txn = None 
                raise

    def _close_db(self):
        """Close the LMDB database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
            self._txn = None
            logger.debug("Closed LMDB connection")

    def _get_length(self):
        """Get the number of samples in the dataset."""
        if self._length is None:
            # Ensure DB is ready before accessing length
            self._open_db_if_needed()
            if self._txn is None:
                 logger.error("LMDB transaction not available when trying to get length. Cannot proceed.")
                 # Decide on behavior: raise error or return 0/None?
                 # Raising error is safer to prevent downstream issues.
                 raise RuntimeError("LMDB transaction required but not available to get dataset length.")

            # Get length from metadata or count entries
            if self._metadata and 'num_samples' in self._metadata:
                self._length = self._metadata['num_samples']
            else:
                # Count entries by accessing the __len__ key
                length_bytes = self._txn.get(b'__len__')
                if length_bytes is not None:
                    self._length = int(length_bytes.decode('ascii'))
                else:
                    # Fall back to counting keys (slower)
                    self._length = sum(1 for _ in self._txn.cursor()) - 1  # exclude __len__
                    logger.warning(f"Had to count {self._length} keys manually. Dataset should store length metadata.")
        return self._length

    def _load_metadata(self):
        """Load the dataset metadata."""
        if self._metadata is None:
            # Ensure DB is ready before accessing metadata
            self._open_db_if_needed()
            if self._txn is None:
                logger.error("LMDB transaction not available when trying to load metadata. Cannot proceed.")
                raise RuntimeError("LMDB transaction required but not available to load metadata.")
            
            if self._metadata_path.exists():
                try:
                    with open(self._metadata_path, 'r') as f:
                        self._metadata = json.load(f)
                    logger.debug(f"Loaded metadata from {self._metadata_path}")
                    
                    # Create gene-to-indices mapping for efficient gene_mask generation
                    if 'genes' in self._metadata:
                        genes = self._metadata['genes']
                        self._gene_idx_map = {}
                        
                        # Load gene-to-indices mapping if it exists in the LMDB
                        gene_idx_bytes = self._txn.get(b'__gene_idx_map__')
                        if gene_idx_bytes is not None:
                            self._gene_idx_map = pickle.loads(gene_idx_bytes)
                            logger.debug("Loaded gene-to-indices mapping from LMDB")
                except Exception as e:
                    logger.error(f"Failed to load metadata: {e}")
                    self._metadata = {}  # Empty dict to avoid retrying
            else:
                self._metadata = {}  # Empty dict if file doesn't exist
        elif self._metadata is None:
            self._metadata = {}  # Empty dict if file doesn't exist

    def __del__(self):
        """Clean up LMDB resources when the dataset is deleted."""
        self._close_db()

    def len(self) -> int:
        """Returns the number of samples in the dataset."""
        # Make sure length is loaded if it hasn't been already
        if self._length is None:
            self._get_length() 
        # Handle case where length loading failed
        if self._length is None:
            logger.error("Dataset length could not be determined.")
            return 0 # Or raise an error
        return self._length

    def get(self, idx: int) -> Data:
        """
        Get a single graph by index.
        
        Args:
            idx: The index of the graph to retrieve
            
        Returns:
            A PyTorch Geometric Data object
        """
        # We need length check here, which implicitly calls _get_length -> _open_db_if_needed
        if not 0 <= idx < len(self):
            # Note: len(self) ensures DB is open and length is loaded
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        # At this point, _open_db_if_needed should have been called by len()
        # but double-check transaction just in case len() failed silently or returned 0
        if self._txn is None:
            logger.error(f"LMDB transaction is None in get() for index {idx}. Trying to reopen.")
            self._open_db_if_needed() # Attempt to reopen
            if self._txn is None:
                raise RuntimeError(f"LMDB transaction failed to initialize in worker for get(idx={idx}).")

        # Convert index to bytes key
        key = f"{idx}".encode('ascii')
        
        # Get the data from LMDB
        data_bytes = self._txn.get(key)
        if data_bytes is None:
            logger.error(f"KeyError in worker {self.worker_id}: Index {idx} (key: {key}) not found in LMDB transaction.")
            raise KeyError(f"Sample with index {idx} not found in LMDB (Worker: {self.worker_id})")
        
        # Deserialize the data
        try:
            # Use BytesIO for more efficient deserialization
            buffer = io.BytesIO(data_bytes)
            data = torch.load(buffer)
        except Exception as e:
            logger.error(f"Failed to deserialize data for index {idx}: {e}")
            raise
        
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
            
        return data

    def get_gene_mask(self, gene: str) -> torch.Tensor:
        """
        Get a boolean mask for all samples belonging to a specific gene.
        
        Args:
            gene: The gene to filter by
            
        Returns:
            Boolean tensor mask where True indicates samples with the specified gene
        """
        # If we have a precomputed gene-to-indices mapping, use it
        if self._gene_idx_map and gene in self._gene_idx_map:
            # Create a boolean mask
            mask = torch.zeros(len(self), dtype=torch.bool)
            # Set indices for this gene to True
            indices = self._gene_idx_map[gene]
            mask[indices] = True
            return mask
        else:
            # Fall back to checking each sample (inefficient)
            logger.warning(f"Generating gene mask for '{gene}' by scanning all samples (slow)")
            mask = []
            for i in range(len(self)):
                # Get data
                data = self.get(i)
                # Check if gene matches
                is_match = data.gene == gene if hasattr(data, 'gene') else False
                mask.append(is_match)
            return torch.tensor(mask, dtype=torch.bool)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the dataset metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    @property
    def num_classes(self):
        """Returns the number of classes (if applicable)."""
        if self.metadata and 'genes' in self.metadata:
            return len(self.metadata['genes'])
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

    # This __getstate__ and __setstate__ is crucial for use with DataLoader workers
    # It ensures that the LMDB environment is not pickled (which causes errors)
    # and is instead reopened in the worker process.
    def __getstate__(self):
        # Return dictionary with LMDB-related attributes removed
        state = self.__dict__.copy()
        state['_db'] = None
        state['_txn'] = None
        state['worker_id'] = None # Reset worker ID
        # Keep essential config needed to reopen
        # state['_db_path'] is already derived from root_dir
        # state['_readonly'], state['_map_size'], state['root_dir'] are preserved
        return state

    def __setstate__(self, state):
        # Restore state
        self.__dict__.update(state)
        # LMDB will be reopened when needed by _open_db_if_needed
        # logger.debug(f"Restored state in PID {os.getpid()}, DB will reopen on first access.") # Optional debug log


# Helper function for multiprocessing conversion
def _worker_serialize_batch(args):
    """Worker function to serialize a batch of samples to LMDB.
    
    No longer calculates global indices; instead returns original keys for lookup
    in the main process against the index.pt data.
    """
    # Unpack the two arguments being passed
    batch_file, lmdb_dir = args
    
    try:
        # Load the batch
        batch_data = torch.load(batch_file)
        
        # Process each sample
        results = []
        for orig_key, data in batch_data.items():
            # Get gene if available
            gene = data.gene if hasattr(data, 'gene') else None
            
            # Create a serialized version for LMDB
            buffer = io.BytesIO()
            torch.save(data, buffer)
            serialized = buffer.getvalue()
            
            # Return the original key for lookup in main process
            # Format: (original_key, serialized_data, gene, batch_file_name)
            # Main process will look up the global index using original_key
            results.append((orig_key, serialized, gene, str(batch_file)))
        
        logger.debug(f"Processed {len(results)} samples from {batch_file}")
        return results
    except Exception as e:
        logger.error(f"Error processing batch {batch_file}: {e}")
        return [] # Return empty list on failure 