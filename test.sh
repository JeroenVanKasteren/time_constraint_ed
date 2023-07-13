#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#time >1 minute
#SBATCH --time 0-00:00:30
#SBATCH --error results/read/prints_%A_%a.err

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

python -u Train.py --id $SLURM_JOBID --index $SLURM_ARRAY_TASK_ID --time $TIME --method 'VI' > out_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.txt

