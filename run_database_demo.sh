#!/bin/bash

# Run the setup script to create the directory structure
echo "Setting up database directory structure..."
python setup_database.py

# Check if data/database directory exists
if [ ! -d "data/database" ]; then
    echo "Error: Failed to create database directory"
    exit 1
fi

# Copy the example file to be used as the database
echo "Setting up example database file..."
cp data/database/example.hpoa data/database/phenotype.hpoa

# Run the demonstration script
echo "Running database demonstration..."
python -m src.database.test_gene_phenotype_db

echo ""
echo "Database demo completed!"
echo ""
echo "You can now use the CLI to query the database:"
echo "python -m src.database.cli gene-phenotypes AARS1"
echo "python -m src.database.cli phenotype-genes HP:0001939"
echo "python -m src.database.cli frequency AARS1 HP:0002460 --disease-id OMIM:613287"
echo "python -m src.database.cli gene-diseases AARS1"
echo "python -m src.database.cli disease-genes OMIM:613287"
echo ""
echo "For help, run: python -m src.database.cli --help" 