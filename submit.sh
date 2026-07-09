#!/bin/bash
#SBATCH --job-name=wb5_nphc
#SBATCH --partition=main
#SBATCH --output=logs/wb5_%A_%a.out  # %A is Job ID, %a is Array Task ID
#SBATCH --error=logs/wb5_%A_%a.err
#SBATCH --array=0-39                  # Spawns 40 parallel tasks (Workers 0 through 39)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1             # 1 CPU per worker
#SBATCH --mem=2G                      # 2 GB RAM per worker is plenty for PHCpack
#SBATCH --time=01:00:00               # 1 hour max time

# Create a logs directory if it doesn't exist
mkdir -p logs

# Execute the python script
# The environment variables SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT 
# are automatically passed into your Python script by SLURM!
python run_nphc_wb5_copy.py