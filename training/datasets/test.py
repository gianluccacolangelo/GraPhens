import torch
import time
import logging
import os
from torch_geometric.loader import DataLoader as PyGDataLoader
import lmdb
import io # Import io for BytesIO

# Assume LMDBHPOGraphDataset is defined in this file or imported
# from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
# Using the placeholder again for context, replace with actual import
try:
    from training.datasets.lmdb_hpo_dataset import LMDBHPOGraphDataset
except ImportError:
    logging.warning("LMDBHPOGraphDataset not found, using placeholder.")
    # Placeholder definition (same as before)
    class LMDBHPOGraphDataset:
        def __init__(self, root_dir, readonly=True, map_size=0):
            self.root_dir = root_dir
            # Simulate finding some indices for dummy genes
            self._gene_idx_map = {
                'GENE_A': list(range(0, 10000, 2)),
                'GENE_B': list(range(1, 10000, 2))
            }
            self._length = 10000 # Simulate a larger dataset
            logging.info(f"Placeholder Dataset initialized at {root_dir} with {self._length} samples")

        def __len__(self):
            return self._length

        def get_gene_mask(self, gene: str) -> torch.Tensor:
            mask = torch.zeros(self._length, dtype=torch.bool)
            if gene in self._gene_idx_map:
                indices = self._gene_idx_map[gene]
                if indices:
                     mask[indices] = True
            return mask

        def get(self, idx):
            from torch_geometric.data import Data
            # Simulate finding gene (inefficiently)
            gene = 'UNKNOWN'
            for g, indices in self._gene_idx_map.items():
                if idx in indices:
                    gene = g
                    break
            # Simulate data loading - adding a tiny sleep to mimic some work
            # time.sleep(0.0001)
            return Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1, 2, 3],[1, 2, 3, 4]]), gene=gene, id=idx)

        def __repr__(self):
            return f'PlaceholderLMDBHPOGraphDataset({len(self)})'


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# IMPORTANT: Set this to the actual root directory of your dataset
DATASET_ROOT_DIR = "/home/gcolangelo/GraPhens/data/simulation/output" # Using path from your log output
# DATASET_ROOT_DIR = "/path/to/your/hpo_dataset" # <--- Or change manually if needed

BATCH_SIZE = 32# Adjust as needed
NUM_WORKERS = 10   # Set based on your CPU cores (e.g., os.cpu_count() or a fixed number)
# NUM_WORKERS = os.cpu_count() # Example: Use all available cores

# Number of batches to test (set to None to iterate through the whole dataset once)
NUM_TEST_BATCHES = 250 # Reduced for faster testing before inspection
# NUM_TEST_BATCHES = 200 # Or None for full epoch


# --- Main Speed Test Logic ---
if __name__ == "__main__":
    logger.info("--- Starting LMDB Dataset Speed Test ---")
    logger.info(f"Dataset Root: {DATASET_ROOT_DIR}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Num Workers: {NUM_WORKERS}")
    logger.info(f"Testing Batches: {'Full Epoch' if NUM_TEST_BATCHES is None else NUM_TEST_BATCHES}")

    # 1. Instantiate the Dataset
    try:
        # Use readonly=True for loading data
        dataset = LMDBHPOGraphDataset(root_dir=DATASET_ROOT_DIR, readonly=True)
        logger.info(f"Successfully loaded dataset: {dataset} with {len(dataset)} samples.")
        if len(dataset) == 0:
             logger.error("Dataset is empty. Cannot run speed test.")
             # Continue to inspection part even if dataset is empty or fails loading
             # exit(1)
    except FileNotFoundError:
        logger.error(f"LMDB database not found at {os.path.join(DATASET_ROOT_DIR, 'hpo_lmdb')}. "
                     "Run conversion script first or check path.")
        # Continue to inspection part
        # exit(1)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        # Continue to inspection part
        # exit(1)

    # 2. Instantiate the PyTorch Geometric DataLoader with standard shuffling
    # Check if dataset was loaded successfully before creating loader
    if 'dataset' in locals() and len(dataset) > 0:
        data_loader = PyGDataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,  # Enable standard random shuffling
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False, # Good practice if using GPU
            persistent_workers=True if NUM_WORKERS > 0 else False, # Can speed up epoch starts
        )

        logger.info("DataLoader created. Starting iteration...")

        # 3. Iterate and Time
        start_time = time.time()
        batches_processed = 0
        samples_processed = 0
        test_successful = True

        try:
            # Using enumerate to easily limit batches if needed
            for i, batch in enumerate(data_loader):
                # The main work being timed is the fetching and collation done by the DataLoader
                # You could add a minimal operation here if needed, like moving to GPU:
                # if torch.cuda.is_available():
                #     batch = batch.to('cuda')

                batches_processed += 1
                samples_processed += batch.num_graphs # PyG batch object tracks graphs

                if NUM_TEST_BATCHES is not None and batches_processed >= NUM_TEST_BATCHES:
                    logger.info(f"Reached target number of batches ({NUM_TEST_BATCHES}). Stopping iteration.")
                    break

                # Optional: Log progress every N batches
                # if batches_processed % 50 == 0:
                #    logger.info(f"Processed {batches_processed} batches...")
        except Exception as e:
             logger.error(f"!!! Error occurred during DataLoader iteration: {e}", exc_info=True)
             logger.error("This likely confirms the indexing issue.")
             test_successful = False


        end_time = time.time()
        total_time = end_time - start_time

        # 4. Report Results
        logger.info("--- Speed Test Results ---")
        if batches_processed > 0 and test_successful:
            avg_time_per_batch = total_time / batches_processed
            samples_per_second = samples_processed / total_time
            logger.info(f"Processed {batches_processed} batches ({samples_processed} samples).")
            logger.info(f"Total Time: {total_time:.3f} seconds")
            logger.info(f"Average Time per Batch: {avg_time_per_batch:.4f} seconds")
            logger.info(f"Samples per Second: {samples_per_second:.2f}")
        elif not test_successful:
             logger.warning("Speed test incomplete due to errors during iteration.")
        else:
            logger.warning("No batches were processed. Check dataset and configuration.")
    else:
        logger.warning("Skipping DataLoader speed test because dataset failed to load or was empty.")

    # --- Direct LMDB Inspection ---
    logger.info("--- Starting Direct LMDB Inspection ---")
    LMDB_PATH = os.path.join(DATASET_ROOT_DIR, "hpo_lmdb")
    INDICES_TO_INSPECT = [0, 1, 10,4999,5000,7000,7001,12500,13500, 50400,6000,6001,100000,1000000] # Small indices, potentially batch indices
    # Also check one larger index that *should* work if keys are sample indices
    # We know 5M failed, let's try something smaller but > 5000


    try:
        env = lmdb.open(LMDB_PATH, readonly=True, max_dbs=1, lock=False)
        logger.info(f"Opened LMDB at {LMDB_PATH} for direct inspection.")

        with env.begin() as txn:
            for index_to_check in INDICES_TO_INSPECT:
                key_bytes = f"{index_to_check}".encode('ascii')
                logger.info(f"Attempting to retrieve key: '{key_bytes.decode()}'")
                value_bytes = txn.get(key_bytes)

                if value_bytes:
                    logger.info(f"  SUCCESS! Found key '{key_bytes.decode()}'. Value length: {len(value_bytes)} bytes.")
                    try:
                        # Attempt to deserialize
                        # import io # Already imported above
                        buffer = io.BytesIO(value_bytes)
                        deserialized_data = torch.load(buffer)

                        # Inspect the type and structure
                        data_type = type(deserialized_data)
                        logger.info(f"  Deserialized data type: {data_type}")

                        if isinstance(deserialized_data, dict):
                            logger.warning(f"  >>> Data is a DICTIONARY (potential batch!). Number of keys: {len(deserialized_data)}")
                            # Check type of first item in dict
                            if deserialized_data:
                                first_item_key = next(iter(deserialized_data))
                                first_item = deserialized_data[first_item_key]
                                logger.info(f"  Type of first item in dict: {type(first_item)}")
                                if hasattr(first_item, 'x'): # Check if it looks like PyG Data
                                     logger.info(f"  First item appears to be a PyG Data object.")

                        elif isinstance(deserialized_data, list):
                             logger.warning(f"  >>> Data is a LIST (potential batch!). Length: {len(deserialized_data)}")
                             # Check type of first item
                             if deserialized_data:
                                 logger.info(f"  Type of first item in list: {type(deserialized_data[0])}")
                                 if hasattr(deserialized_data[0], 'x'):
                                      logger.info(f"  First item appears to be a PyG Data object.")

                        elif hasattr(deserialized_data, 'x') and hasattr(deserialized_data, 'edge_index'):
                             logger.info(f"  Data appears to be a single PyG Data object (as expected for individual samples).")
                        else:
                            logger.info(f"  Data is of an unrecognized type/structure for this check.")

                        # --- Added: Detailed Inspection if it's a Data object ---
                        if index_to_check == 0 and hasattr(deserialized_data, 'x'): # Inspect the first item if it looks like PyG Data
                            logger.info(f"  --- Detailed Inspection of Data object (Index 0) ---")
                            try:
                                for key, item in deserialized_data:
                                    if torch.is_tensor(item):
                                        logger.info(f"    {key}: Tensor(shape={item.shape}, dtype={item.dtype})")
                                    else:
                                        logger.info(f"    {key}: {item} (type={type(item)})")
                                logger.info(f"  --- End Detailed Inspection ---")
                            except Exception as inspect_e:
                                 logger.error(f"    Failed during detailed inspection: {inspect_e}")
                        # --- End Added ---   

                    except Exception as e:
                        logger.error(f"  FAILED to deserialize or inspect data for key '{key_bytes.decode()}': {e}")
                else:
                    logger.error(f"  FAILED! Key '{key_bytes.decode()}' not found.")

        env.close()
        logger.info("Closed LMDB environment for direct inspection.")

    except FileNotFoundError:
        logger.error(f"LMDB database environment not found at directory: {LMDB_PATH}. Check the path.")
    except Exception as e:
        logger.error(f"An error occurred during direct LMDB inspection: {e}", exc_info=True)

    # --- Previous LMDB Length Check Code (kept for reference) ---
    # Check LMDB length
    # LMDB_PATH = os.path.join(DATASET_ROOT_DIR, "hpo_lmdb")
    # try:
    #     # Get reported length from dataset class
    #     reported_len = len(dataset)
    #     logger.info(f"Reported dataset length (__len__): {reported_len}")

    #     # Get length stored in LMDB explicitly
    #     env = lmdb.open(LMDB_PATH, readonly=True, max_dbs=1)
    #     with env.begin() as txn:

    #         len_bytes = txn.get(b'__len__')
    #         stored_len = int(len_bytes.decode('ascii')) if len_bytes else "Not Found"
    #         logger.info(f"Length stored in LMDB key '__len__': {stored_len}")

    #         # Count actual data keys (can be slow for very large DBs)
    #         logger.info("Counting actual data keys in LMDB (this might take a while)...")
    #         actual_count = 0
    #         cursor = txn.cursor()
    #         for key, _ in cursor:
    #              # Exclude metadata keys
    #              if not key.startswith(b'__'):
    #                  actual_count += 1
    #         logger.info(f"Actual number of data keys found: {actual_count}")

    #     env.close()

    #     if reported_len != actual_count:
    #          logger.warning(f"MISMATCH! Reported length ({reported_len}) does not match actual key count ({actual_count}).")
    #     if isinstance(stored_len, int) and stored_len != actual_count: # Check if stored_len is int before comparing
    #          logger.warning(f"MISMATCH! Stored length key ({stored_len}) does not match actual key count ({actual_count}).")


    # except FileNotFoundError:
    #     logger.error(f"LMDB not found at {LMDB_PATH}")
    # except Exception as e:
    #     logger.error(f"An error occurred: {e}")
