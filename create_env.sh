#In case creating takes too long, create with a job script
#!/bin/bash
#Set job requirements
#SBATCH --job-name="create_env"
#SBATCH -t 00:5:00
#SBATCH --output=init_env.out

#load modules
module load 2022
module load Anaconda3/2022.05

echo "Start updating conda"
conda init
#Conda update conda
conda update conda
echo "Create env"
time conda env create -f environment.yml
conda activate time_constraint_ed
