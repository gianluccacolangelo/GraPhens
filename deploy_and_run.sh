#!/bin/bash
# deploy_and_run.sh - Deploy your code to the cluster and then run it on the best node.
#
# Usage examples:
#   ./deploy_and_run.sh                     # Runs the default command ("ls")
#   ./deploy_and_run.sh my_script.py nodo3            # Runs "my_script.py" on nodo3
#   ./deploy_and_run.sh "python my_script.py" nodo8 gpu=2 cpu=10  # Runs with 2 GPUs, 10 CPUs on nodo8
#
# This script:
# 1. Calls deploy.sh to update your code on the cluster.
# 2. SSHes into the cluster and uses srun to request an interactive session on the best node.
#    The best node is selected by the BEST_NODE variable (e.g. "nodo3").
# 3. In that session, it changes to your project directory, prints diagnostic info
#    (like nvidia-smi and the working directory), activates the conda environment,
#    and runs the specified command.
#
# Note: The srun warnings regarding pseudo-terminal allocation are usually harmless.
#       If you need a fully interactive terminal, you might remove the --pty flag.
#
# Set variables (customize these paths and node as needed)
REMOTE_USER="gcolangelo"
REMOTE_HOST="cluster"
REMOTE_PROJECT_DIR="/home/gcolangelo/GraPhens/"

# Determine the command to run dynamically.
# If parameters are provided, they form the RUN_COMMAND; otherwise, default to "ls".
if [ $# -ge 1 ]; then
    RUN_COMMAND="$1"
else
    RUN_COMMAND="ls"
fi

# Set node if provided, otherwise default to nodo3
if [ $# -ge 2 ]; then
    BEST_NODE="$2"
else
    BEST_NODE="nodo3"
fi

# Default values
GPU_COUNT=0
CPU_COUNT=8
PARTITION="cpu"

# Process remaining parameters
for param in "${@:3}"; do
    # Extract GPU count
    if [[ $param =~ gpu=([0-9]+) ]]; then
        GPU_COUNT="${BASH_REMATCH[1]}"
        echo "Using $GPU_COUNT GPU(s)"
    fi
    
    # Extract CPU count
    if [[ $param =~ cpu=([0-9]+) ]]; then
        CPU_COUNT="${BASH_REMATCH[1]}"
        echo "Using $CPU_COUNT CPU(s)"
    fi
done

# Configure GPU parameters for srun
GPU_OPTION=""
if [ $GPU_COUNT -gt 0 ]; then
    GPU_OPTION="--gres=gpu:$GPU_COUNT"
    PARTITION="gpu"
fi

echo "Starting deployment to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."
./deploy.sh push
echo "Deployment finished."

echo "Starting job on ${BEST_NODE} with command: ${RUN_COMMAND}"
echo "Partition: $PARTITION, CPUs: $CPU_COUNT, GPU option: $GPU_OPTION"

LOGDIR="$HOME/cluster_run_logs"
mkdir -p "$LOGDIR"
echo "$(date +'%F %T') node=$BEST_NODE gpus=$GPU_COUNT cpus=$CPU_COUNT cmd=$RUN_COMMAND" \\
     >> "$LOGDIR/history.log"

# The srun command below forces allocation on BEST_NODE via --nodelist.
ssh "${REMOTE_HOST}" "srun --nodelist=${BEST_NODE} --nodes=1 --ntasks=1 --cpus-per-task=$CPU_COUNT -p $PARTITION $GPU_OPTION --time=140:00:00 --pty bash -c 'cd ${REMOTE_PROJECT_DIR} && echo \"Running on node: \$(hostname)\" && echo \"SLURM Node List: \$SLURM_JOB_NODELIST\" && echo \"Current working directory: \$(pwd)\" && if [ $GPU_COUNT -gt 0 ]; then nvidia-smi; fi && source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate graphens && ${RUN_COMMAND}'"

echo "Pulling results from ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."
./deploy.sh pull
echo "Pulling results finished."

echo "Job completed."
