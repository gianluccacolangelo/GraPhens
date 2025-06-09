#!/bin/bash
# deploy.sh - Deploy your local project to the cluster using rsync or pull files from the cluster.
#
# Usage:
#   ./deploy.sh [push|pull] [max_delete_size]
#     push: Push local files to the cluster (default)
#     pull: Pull files from the cluster to local machine
#     max_delete_size: Optional - Maximum size in MB for files to be deleted (e.g., 100M)
#                      Files larger than this size won't be deleted during sync
#
# Requirements:
# - Your ~/.ssh/config must define a host alias (e.g., "cluster").
# - The remote destination directory must exist (or you can modify the script to create it).

# Variables (change these as needed)
REMOTE_USER="gcolangelo"
REMOTE_HOST="cluster"
REMOTE_PROJECT_DIR="/home/gcolangelo/GraPhens"

# Default operation is push
OPERATION=${1:-"push"}
# Default max size for deletion is 400MB (consistent with --max-size)
MAX_DELETE_SIZE=${2:-"400M"}

# Temporary file for exclusions
EXCLUDE_FILE=$(mktemp)
trap "rm -f $EXCLUDE_FILE" EXIT

if [ "$OPERATION" = "push" ]; then
    echo "Deploying code to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."
    
    # Generate exclusion list for files larger than MAX_DELETE_SIZE
    echo "Generating list of large files to protect from deletion (larger than $MAX_DELETE_SIZE)..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "find ${REMOTE_PROJECT_DIR} -type f -size +${MAX_DELETE_SIZE} | sed 's|${REMOTE_PROJECT_DIR}/||'" > "$EXCLUDE_FILE"
    echo "Found $(wc -l < "$EXCLUDE_FILE") files to protect from deletion. Those files are: "
    cat "$EXCLUDE_FILE"
    
    # Sync local directory to the remote project directory
    rsync -avz --delete --max-size=400M --exclude '.git/' --exclude '*.pt' --exclude '*.mdb' --exclude 'data/simulation/output/simulated_patients_all_*.json' --exclude-from="$EXCLUDE_FILE" . "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
    
    echo "Deployment complete."
elif [ "$OPERATION" = "pull" ]; then
    echo "Pulling files from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR} to local directory..."
    
    # Sync remote directory to the local project directory
    rsync -avz --exclude '.git/' --exclude '*.pt' --max-size=400M "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/" .
    
    echo "Pull complete."
else
    echo "Invalid operation: $OPERATION"
    echo "Usage: ./deploy.sh [push|pull] [max_delete_size]"
    exit 1
fi
