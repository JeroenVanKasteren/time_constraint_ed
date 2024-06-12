#!/bin/bash
#SBATCH --job-name "vi_ED_solve"
#SBATCH --cpus-per-task 16
#SBATCH --time 0-01:00:00
#SBATCH --array 1-10
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

# vi ospi sdf fcfs pi
python train.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance 02 --method vi
# python read_results_sim.py
