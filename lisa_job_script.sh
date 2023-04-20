#Experiment with increasing cpus-per-task
#!/bin/bash
#Set job requirements
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 00:01:00
#SBATCH --output Results/print_$SLURM_JOBID

conda activate time_constraint_ed
#export PYTHONPATH=$PYTHONPATH:$PWD
#Run Train.py, $SLURM_ARRAY_TASK_ID, $HOME/time_constraint_ed/
python Train.py --id $SLURM_ARRAY_TASK_ID --multiplier 42 --J 2 --gamma 30 --policy False --time %t
# python Train.py --id '324125' --multiplier 42 --J 2 --gamma 30 --policy False --time '00:01:00'