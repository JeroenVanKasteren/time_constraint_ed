#!/bin/bash
#SBATCH --job-name "time_constraint_ED"
#SBATCH --nodes 1
#SBATCH --time 0-03:00:00
#SBATCH --array 1-10
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

python train.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance 01 --method vi