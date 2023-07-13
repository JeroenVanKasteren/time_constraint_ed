#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition normal
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#time >1 minute
#SBATCH --time 2-00:00:00
#SBATCH --array 1-10
#SBATCH --output Results/prints_%A_%a.out
#SBATCH --error Results/prints_%A_%a.err

python Train.py --id $SLURM_JOBID --index $SLURM_ARRAY_TASK_ID --J 2 --gamma 25 --policy False --time 1-23:30:00
> results/out_$SLURM_JOBID$SLURM_ARRAY_TASK_ID.out