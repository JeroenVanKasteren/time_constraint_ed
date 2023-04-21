#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#time >1 minute
#SBATCH --time 01:01:00
#SBATCH --array 1-3
#SBATCH --output Results/prints_%A_%a.out
#SBATCH --error Results/prints_%A_%a.err

python Train.py --id $SLURM_JOBID --index $SLURM_ARRAY_TASK_ID --J 2 --gamma 30 --policy False --time "01:00:30"
