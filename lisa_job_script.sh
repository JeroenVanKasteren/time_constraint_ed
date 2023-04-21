#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#time >1 minute
#SBATCH --time 00:03:00
#SBATCH --array 1-1
#SBATCH --output Results/prints_%A_%a.out
#SBATCH --error Results/prints_%A_%a.err

# conda activate time_constraint_ed
# Run Train.py
python Train.py --id $SLURM_ARRAY_TASK_ID --multiplier 42 --J 2 --gsqueueamma 30 --policy False --time "00:03:00"
