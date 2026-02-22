import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phenotypes_per_gene_distribution(input_file, output_file):
    """
    Reads phenotype-to-gene mappings and plots the distribution of 
    number of phenotypes per gene.
    """
    print(f"Reading data from {input_file}...")
    try:
        # Assuming tab-separated based on the header structure
        # low_memory=False to handle mixed types in columns not used
        df = pd.read_csv(input_file, sep='\t', dtype={'ncbi_gene_id': str, 'disease_id': str})
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check required columns
    required_cols = ['hpo_id', 'gene_symbol']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file must contain columns: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Group by gene and count unique phenotypes
    print("Calculating phenotypes per gene...")
    # distinct phenotypes per gene
    phenotypes_per_gene = df.groupby('gene_symbol')['hpo_id'].nunique()
    
    # Calculate statistics
    stats = phenotypes_per_gene.describe()
    print("\nSummary Statistics (Phenotypes per Gene):")
    print(stats)
    
    # Plotting
    print(f"Generating plot to {output_file}...")
    plt.figure(figsize=(8, 8))
    sns.set_style("white")
    
    # Histogram without KDE
    sns.histplot(phenotypes_per_gene, kde=False, bins=50, legend=False, color='#808080', alpha=1)
    
    # Remove title and labels
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    
    # Increase tick parameters significantly (approx 20x visual weight now)
    plt.tick_params(axis='both', which='major', labelsize=60, width=8, length=20)
    
    # Make axis spines thicker
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(8)
    
    # Remove grids
    plt.grid(False)
    sns.despine()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)
    print("Done.")

if __name__ == "__main__":
    INPUT_FILE = "data/phenotype_to_genes.txt"
    OUTPUT_FILE = "gene_phenotype_distribution.png"
    
    if os.path.exists(INPUT_FILE):
        plot_phenotypes_per_gene_distribution(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"File not found: {INPUT_FILE}")

