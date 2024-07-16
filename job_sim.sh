#!/bin/bash
#SBATCH --job-name "ED_sim"
#SBATCH --cpus-per-task 16
#SBATCH --time 0-01:00:00
#SBATCH --array 1-107
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

# instance: J1 J2 J2_D_gam, J3, sim
python run_simulations.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance J2 --max_iter 1e6 --continue_run True
# python read_results_sim.py