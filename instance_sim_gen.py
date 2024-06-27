"""
Creation of instance file to simulate the TimeConstraintEDs problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 31-5-2023.
"""

import numpy as np
import os
import pandas as pd
from utils import instances_sim

FILEPATH_INSTANCE = 'results/sim_instances_01.csv'

# sim_ids = list(map("{:02d}".format, list(range(1, 15))))
# for sim_id in sim_ids:
#     FILEPATH_INSTANCE = 'results/instance_sim_' + str(sim_id) + '.csv'
methods = ['ospi',  'cmu_t_min', 'cmu_t_max', 'fcfs', 'sdf', 'sdfprior',
           'l_max', 'l_min']
instance_columns = ['J', 'S', 'gamma', 'D',
                    'mu', 'lab', 'load', 'imbalance'
                    't', 'c', 'r',
                    'e', 'P',
                    'max_t_prob', 'N', 'start_K', 'batch_T']

method_columns = ['_job_id', '_attempts', '_time', '_iter',
                  '_g', '_g_ci', '_perc', '_perc_ci']
for method in methods:
    instance_columns.extend([method + s for s in method_columns])

inst = pd.DataFrame(0, index=np.arange(len(methods)),
                    columns=instance_columns)
inst = instances_sim.generate_instance(inst, int(sim_id))
inst['method'] = methods
inst = inst[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file with this instance already exists, name: ',
          FILEPATH_INSTANCE)
else:
    inst.to_csv(FILEPATH_INSTANCE)
