import json

def count_cases_per_gene(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        gene_case_counts = {}

        for gene, cases in data.items():
            gene_case_counts[gene] = len(cases)

        for gene, count in gene_case_counts.items():
            print(f"Gene: {gene}, Cases: {count}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = 'data/simulation/output/simulated_patients_all_20250331_141155.json'
    count_cases_per_gene(file_path)

