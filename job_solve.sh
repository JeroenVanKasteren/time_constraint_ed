#!/bin/bash
#SBATCH --job-name "vi_ED_solve"
#SBATCH --cpus-per-task 16
#SBATCH --time 0-04:00:00
#SBATCH --array 1-1
#SBATCH --output results/read/prints_%A_%a.out
#SBATCH --error results/read/prints_%A_%a.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

# instance: J1 J2 J2_D_gam, J3
# method: vi ospi sdf fcfs pi cmu_t_min
# python train.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance J2 --method vi
python read_results.py --instance J1
python read_results.py --instance J2
python read_results.py --instance J2_D_gam
