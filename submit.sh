#!/bin/bash
#SBATCH --job-name=wb5_nphc_max
#SBATCH --partition=main
#SBATCH --output=logs/%j_wb5_max.out
#SBATCH --error=logs/%j_wb5_max.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # Request ALL 16 CPU cores on the node!
#SBATCH --mem=8G                      # 8 GB RAM to support 16 simultaneous solvers
#SBATCH --time=01:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the auto-scaling script
# SLURM automatically passes SLURM_CPUS_PER_TASK=16 to Python
python run_nphc_wb5_copy.py