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

# 1. Enter the directory where the sbatch command was executed
cd $SLURM_SUBMIT_DIR

# 2. Activate Conda cleanly
source /software/conda/etc/profile.d/conda.sh
conda activate pytorch

# =================================================================
# THE CRITICAL C++ BRIDGE (Now safely tracked in Git!):
# Tells Linux to use your Conda env's modern C++ library so libPHCpack
# doesn't crash with a NoneType error on the compute node.
# =================================================================
export LD_LIBRARY_PATH=/home/g202210120/.conda/envs/pytorch/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/home/g202210120/.conda/envs/pytorch/lib/libstdc++.so.6

# 3. Create logs folder if it doesn't exist
mkdir -p logs

# 4. Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_PATH=$(which python)

# 5. Execute the solver
if command -v numactl &> /dev/null; then
    echo "Running with NUMA memory interleaving enabled using: $PYTHON_PATH"
    numactl --interleave=all "$PYTHON_PATH" run_nphc_wb5_copy.py
else
    echo "numactl not found, running standard python execution..."
    python run_nphc_wb5_copy.py
fi