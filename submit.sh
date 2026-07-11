#!/bin/bash
#SBATCH --job-name=wb5_nphc_max
#SBATCH --partition=cpu_x440                 
#SBATCH --output=logs/%j_wb5.out         
#SBATCH --error=logs/%j_wb5.err          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                   
#SBATCH --mem=64G                            
#SBATCH --time=01:00:00

# =================================================================
# FIX: Force the compute node to enter your project directory!
# =================================================================
cd $SLURM_SUBMIT_DIR

# =================================================================
# Activate Conda
# =================================================================
source /software/conda/etc/profile.d/conda.sh
conda activate pytorch
export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD
# Create logs directory if it doesn't exist
mkdir -p logs

# Force math libraries to stay strictly single-threaded per worker process.
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