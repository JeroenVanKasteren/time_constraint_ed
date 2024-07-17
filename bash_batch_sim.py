from subprocess import run

JOBSCRIPT = """#!/bin/bash
#SBATCH --job-name {name}
#SBATCH --cpus-per-task 16
#SBATCH --time {time}
#SBATCH --array 1-{n_array}
#SBATCH --output results/read/prints_%A_%a_{name}.out
#SBATCH --error results/read/prints_%A_%a_{name}.err

# time is rounded to minutes
TIMELIMIT=`squeue -j $SLURM_JOB_ID -o "%l" | tail -1`

python run_simulations.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance {instance} --max_iter 1e6 --continue_run True
# python read_results_sim.py
"""

instances = [['J1', '0-00:20:00', '108'],
             ['J2', '0-00:20:00', '108'],
             ['sim', '0-00:20:00', '10']]

for instance, time, n_array in instances:
    script = JOBSCRIPT.format(
        name=instance + '_sim_ED',
        time=time,
        n_array=n_array,
        instance=instance)

    run(["sbatch"], input=script.encode())
    # print(script.encode())
