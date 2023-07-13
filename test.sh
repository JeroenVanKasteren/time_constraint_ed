#!/bin/bash
#Set job requirements & Experiment with increasing cpus-per-task
#SBATCH --job-name "time_constraint_ED"
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#time >1 minute
#SBATCH --time 0-00:00:30
#SBATCH --array 1-1
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

python test.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIME --method OSPI > results/out_$SLURM_JOBID$SLURM_ARRAY_TASK_ID.out
