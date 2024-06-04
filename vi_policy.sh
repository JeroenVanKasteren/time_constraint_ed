#!/bin/bash
#SBATCH --job-name "time_constraint_ED_pol"
#SBATCH --cpus-per-task 8
#SBATCH --time 0-00:10:00
#SBATCH --array 1-107
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

python test_other/Sandbox.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance 01 --x 1e5 --method vi
