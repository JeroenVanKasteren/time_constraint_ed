#In case creating takes too long, create with a job script
# #!/bin/bash
# #Set job requirements
# #SBATCH --job-name="create_env"
# #SBATCH -t 00:5:00
# #SBATCH --output=init_env.out

#load modules
module load 2022
module load Anaconda3/2022.05
conda create -n time_constraint_ed python=3.10.10 anaconda
conda activate time_constraint_ed
conda env update --name time_constraint_ed -file requirements.txt
# or
# pip install -r requirements.txt
