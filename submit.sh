#!/bin/bash
#SBATCH --job-name=wb5_nphc_max
#SBATCH --partition=cpu_x440                 # Target the idle cpu_x440 partition
#SBATCH --output=logs/%j_wb5.out         # Log filename starts with Job ID
#SBATCH --error=logs/%j_wb5.err          # Error log starts with Job ID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                   # Request 64 physical execution cores
#SBATCH --mem=64G                            # 64 GB RAM (1 GB per solver process)
#SBATCH --time=01:00:00

# =================================================================
# FIX: explicitly activate Conda on the remote compute node!
# =================================================================
source /software/conda/etc/profile.d/conda.sh
conda activate pytorch

# Create logs directory if it doesn't exist
mkdir -p logs

# Force math libraries to stay strictly single-threaded per worker process.
# This prevents 64 Python workers from attempting to spawn 64 threads each!
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Get the absolute path of the currently active Python interpreter
PYTHON_PATH=$(which python)

# numactl --interleave=all stripes RAM access evenly across all NUMA nodes
if command -v numactl &> /dev/null; then
    echo "Running with NUMA memory interleaving enabled using: $PYTHON_PATH"
    numactl --interleave=all "$PYTHON_PATH" run_nphc_wb5_warmstart_gemini.py
else
    echo "numactl not found, running standard python execution..."
    python run_nphc_wb5_warmstart_gemini.py
fi