import subprocess
import os
import logging
import shlex

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_candidate_genes(
    phenotypes: list[str],
    output_dir: str = "phen2gene_output",
    use_docker: bool = True,
    docker_image: str = "genomicslab/phen2gene",
    phen2gene_script: str = "phen2gene.py"
) -> list[dict]:
    """
    Runs Phen2Gene tool with a list of HPO IDs and returns prioritized genes.

    Args:
        phenotypes: A list of HPO ID strings (e.g., ["HP:0000021", "HP:0000027"]).
        output_dir: Directory to store Phen2Gene output files (on the host machine).
        use_docker: If True, run Phen2Gene using the Docker image. Defaults to True.
        docker_image: The name of the Phen2Gene Docker image to use.
        phen2gene_script: Path to the phen2gene.py script (used only if use_docker is False).

    Returns:
        A list of dictionaries, where each dictionary represents a ranked gene
        with keys: 'Rank', 'Gene', 'ID', 'Score', 'Status'.
        Returns an empty list if errors occur or no phenotypes are provided.
    """
    logging.info(f"Starting Phen2Gene analysis for {len(phenotypes)} phenotypes.")

    if not phenotypes:
        logging.warning("No phenotypes provided. Returning empty list.")
        return []

    try:
        # Ensure the output directory exists on the host
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        logging.debug(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir}: {e}")
        return []

    # Construct the command
    if use_docker:
        # The output directory name within the container's mounted volume
        results_dirname = "prioritizedgenelist"
        # Mount the host output directory to /code/out in the container
        volume_mount = f"{abs_output_dir}:/code/out"
        # Set the output path to be relative to the container's working directory
        container_output_path = f"out/{results_dirname}"
        
        cmd = [
            "docker", "run", "--rm",
            "-v", volume_mount,
            docker_image,
            "-m"
        ] + phenotypes + ["-out", container_output_path]
        executor = "Docker"
    else:
        # For local execution, specify the full path with results folder
        results_dirname = "prioritizedgenelist"
        full_output_path = os.path.join(abs_output_dir, results_dirname)
        cmd = ["python3", phen2gene_script, "-m"] + phenotypes + ["-out", full_output_path]
        executor = "python3"

    logging.debug(f"Constructed command: {' '.join(shlex.quote(c) for c in cmd)}")

    try:
        # Execute the command
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
        logging.info(f"Phen2Gene script executed successfully via {executor}.")
        logging.debug(f"Phen2Gene stdout:\n{process.stdout}")
        if process.stderr:
             logging.debug(f"Phen2Gene stderr:\n{process.stderr}")

    except FileNotFoundError:
        logging.error(f"Error: '{cmd[0]}' command not found. Make sure {executor} is installed and in your PATH.")
        if use_docker:
            logging.error("Ensure the Docker daemon is running and you have pulled the image: `docker pull {docker_image}`")
        return []
    except subprocess.CalledProcessError as e:
        logging.error(f"Phen2Gene execution via {executor} failed with exit code {e.returncode}.")
        logging.error(f"Command: {' '.join(shlex.quote(c) for c in cmd)}")
        logging.error(f"Stderr:\n{e.stderr}")
        logging.error(f"Stdout:\n{e.stdout}")
        if use_docker and "permission denied" in e.stderr.lower():
             logging.error("Docker permission error likely. Ensure your user is part of the 'docker' group or run with sudo (not recommended for general use).")
        return []
    except Exception as e:
         logging.error(f"An unexpected error occurred during Phen2Gene execution via {executor}: {e}")
         return []

    # --- Parse the output file ---
    results = []
    
    # Construct the expected output file path based on the Docker command pattern
    if use_docker:
        # When using Docker, output is in the prioritizedgenelist directory inside the mounted volume
        output_file_path = os.path.join(abs_output_dir, results_dirname, "output_file.associated_gene_list")
    else:
        # When running locally, output is in the specified directory path
        output_file_path = os.path.join(abs_output_dir, results_dirname, "output_file.associated_gene_list")
    
    logging.debug(f"Looking for output file at: {output_file_path}")
    
    # Check if output file exists
    if not os.path.exists(output_file_path):
        logging.error(f"Phen2Gene output file not found at: {output_file_path}")
        # List directory contents for debugging
        try:
            logging.debug(f"Contents of {os.path.dirname(output_file_path)}:")
            if os.path.exists(os.path.dirname(output_file_path)):
                files = os.listdir(os.path.dirname(output_file_path))
                logging.debug(f"Files: {files}")
            else:
                logging.debug(f"Directory {os.path.dirname(output_file_path)} does not exist")
                
            # Check parent directory
            parent_dir = os.path.dirname(os.path.dirname(output_file_path))
            logging.debug(f"Contents of parent directory {parent_dir}:")
            if os.path.exists(parent_dir):
                files = os.listdir(parent_dir)
                logging.debug(f"Files: {files}")
        except Exception as e:
            logging.error(f"Error listing directory contents: {e}")
        return []

    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            # Read header, split by whitespace, check consistency
            header_line = f.readline().strip()
            header = header_line.split() # Simple split by whitespace
            expected_fields = ['Rank', 'Gene', 'ID', 'Score', 'Status']

            # Check if header looks like tab-separated
            if len(header) < len(expected_fields) and '\t' in header_line:
                 header = [h.strip() for h in header_line.split('\t')]

            if header != expected_fields:
                 logging.warning(f"Output file header '{header}' might deviate from expected '{expected_fields}'. Attempting to parse anyway.")
                 # We'll proceed assuming the columns are in the expected order.

            # Read data lines
            for i, line in enumerate(f):
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                parts = line.split() # Split by whitespace
                # If whitespace split doesn't yield 5 parts, try tab split
                if len(parts) != 5 and '\t' in line:
                    parts = [p.strip() for p in line.split('\t')]

                if len(parts) != 5:
                     logging.warning(f"Skipping malformed line {i+2} in {output_file_path}: '{line}'")
                     continue
                try:
                    results.append({
                        'Rank': int(parts[0]),
                        'Gene': parts[1],
                        'ID': parts[2],
                        'Score': float(parts[3]),
                        'Status': parts[4]
                    })
                except ValueError as e:
                    logging.warning(f"Skipping row {i+2} in {output_file_path} due to data conversion error: {parts} - {e}")

    except FileNotFoundError:
        logging.error(f"Phen2Gene output file not found: {output_file_path}")
        return []
    except Exception as e:
        logging.error(f"Failed to read or parse output file {output_file_path}: {e}")
        return []

    logging.info(f"Successfully parsed {len(results)} genes from Phen2Gene output.")
    logging.debug(f"Parsed genes: {results[:5]}...") # Log first few results for debugging
    return results

# Example usage with Docker (default)
if __name__ == "__main__":
    phenotypes = ["HP:0001250"] # Example: Seizure
    genes = get_candidate_genes(phenotypes, output_dir="my_phen2gene_results")
    if genes:
        print(f"Found {len(genes)} genes. Top 5:")
        for gene in genes[:5]:
            print(f"Rank: {gene['Rank']}, Gene: {gene['Gene']}, Score: {gene['Score']}, Status: {gene['Status']}")

# Example explicitly asking *not* to use Docker
# (Requires Phen2Gene repo cloned and setup.sh executed)
# genes_local = get_candidate_genes(
#     phenotypes,
#     output_dir="my_phen2gene_results_local",
#     use_docker=False,
#     phen2gene_script="/path/to/your/Phen2Gene/phen2gene.py" # <-- Update this path
# )
# if genes_local:
#      print(genes_local[:5])

# Example usage (optional, can be removed or put under if __name__ == '__main__'):
# if __name__ == '__main__':
#     # Ensure phen2gene.py is in the same directory or provide the correct path
#     # Ensure you have run setup.sh for Phen2Gene first if needed
#     sample_phenotypes = ["HP:0001250"] # Example: Seizure
#     # Using Docker (default)
#     genes = get_candidate_genes(sample_phenotypes, output_dir="phen2gene_docker_output")
#
#     if genes:
#         print("\n--- Top 5 Candidate Genes (Docker) ---")
#         for gene in genes[:5]:
#             print(f"Rank: {gene['Rank']}, Gene: {gene['Gene']}, Score: {gene['Score']:.4f}, Status: {gene['Status']}")
#     else:
#         print("Failed to retrieve candidate genes using Docker.")
#
#     # --- Example using local script (requires Phen2Gene cloned and setup) ---
#     # try:
#     #     genes_local = get_candidate_genes(
#     #         sample_phenotypes,
#     #         output_dir="phen2gene_local_output",
#     #         use_docker=False,
#     #         phen2gene_script="/path/to/cloned/Phen2Gene/phen2gene.py" # IMPORTANT: Update this path
#     #     )
#     #     if genes_local:
#     #         print("\\n--- Top 5 Candidate Genes (Local Script) ---")
#     #         for gene in genes_local[:5]:
#     #             print(f"Rank: {gene['Rank']}, Gene: {gene['Gene']}, Score: {gene['Score']:.4f}, Status: {gene['Status']}")
#     #     else:
#     #         print("Failed to retrieve candidate genes using local script.")
#     # except Exception as e:
#     #      print(f"Error running local example: {e}") # Catch potential path errors etc.
#
#     # Example with multiple phenotypes
#     sample_phenotypes_multi = ["HP:0000021", "HP:0000027", "HP:0030905", "HP:0010628"]
#     genes_multi = get_candidate_genes(sample_phenotypes_multi, output_dir="phen2gene_multi_docker_output")
#     if genes_multi:
#         print("\n--- Top 5 Candidate Genes (Multi) ---")
#         for gene in genes_multi[:5]:
#             print(f"Rank: {gene['Rank']}, Gene: {gene['Gene']}, Score: {gene['Score']:.4f}, Status: {gene['Status']}")
#     else:
#          print("Failed to retrieve candidate genes for multiple phenotypes.") 