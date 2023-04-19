#Experiment with increasing cpus-per-task
#!/bin/bash
#Set job requirements
#SBATCH --job-name "time_constraint_ED"
#SBATCH --partition short
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 0:01:00
#SBATCH --output Results/print_$SLURM_JOBID

#Activate environment
conda activate time_constraint_ed
#export PYTHONPATH=$PYTHONPATH:$PWD

#Run Train.py
python $HOME/time_constraint_ed/src/Train.py --id $SLURM_ARRAY_TASK_ID --multiplier 42 --J 2 --gamma 30 --policy False --time %t
