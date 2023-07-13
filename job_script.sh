#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition normal
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#time >1 minute
#SBATCH --time 2-00:00:00
#SBATCH --array 1-10
#SBATCH --error results/read/prints_%A_%a.err

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

python -u Train.py --id $SLURM_JOBID --index $SLURM_ARRAY_TASK_ID --time $TIME --method 'VI' > out_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.txt

