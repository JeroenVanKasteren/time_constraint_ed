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

python train.py --job_id $SLURM_JOBID --array_id $SLURM_ARRAY_TASK_ID --time $TIMELIMIT --instance {instance} --method {method}
# python read_results_sim.py
"""

methods = ['vi', 'ospi', 'ospi_cons', 'ospi_lin', 'ospi_abs',
           'sdf', 'fcfs', 'pi', 'cmu_t_min']
instances = [['J1', '0-01:00:00', '108', methods],
             ['J2', '0-06:00:00', '108', methods],
             ['J2_D_gam', '0-06:00:00', '216', ['vi', 'ospi']]]

for instance, time, n_array in instances:
    for method in methods:
        script = JOBSCRIPT.format(
            name=instance + '_' + method + '_ED',
            time=time,
            n_array=n_array,
            instance=instance,
            method=method)

        run(["sbatch"], input=script.encode())
        # print(script.encode())
