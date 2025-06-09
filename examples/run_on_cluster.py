#!/usr/bin/env python3
"""
Example demonstrating how to run GraPhens on the cluster.

This script uses the howtorunanything.mdc rule to run the simple_example.py
script on the cluster.
"""

import sys
import os

# Add the parent directory to the path to import GraPhens
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphens import GraPhens

if __name__ == "__main__":
    # Get the path to the simple_example.py script
    script_path = os.path.join(os.path.dirname(__file__), "simple_example.py")
    
    # Run the script on the cluster using node 8
    GraPhens.run_on_cluster(script_path, node="nodo8")
    
    # You can also run other scripts with different nodes:
    # GraPhens.run_on_cluster("path/to/another_script.py", node="nodo9") 