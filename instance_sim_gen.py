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
from utils import TimeConstraintEDs as Env

FILEPATH_INSTANCE = 'results/instance_sim_10.csv'
methods = ['ospi', 'cmu', 'fcfs', 'sdf', 'sdfprior']
input_columns = ['J', 'S', 'gamma', 'D', 't', 'c', 'r', 'mu', 'lab', 'load',
                 'imbalance']
instance_columns = [*input_columns, 'N', 'start_K', 'batch_T',
                    'method', 'g', 'conf_int']

inst = pd.DataFrame(0, index=np.arange(len(methods)), columns=instance_columns)

inst['J'] = 3
inst['S'] = 5
inst['t'] = [np.array([60, 60, 60]) for r in range(len(inst))]
inst['D'] = 180
inst['gamma'] = 1
inst['c'] = [np.array([1, 1, 1]) for r in range(len(inst))]
inst['r'] = [np.array([1, 1, 1]) for r in range(len(inst))]
inst['mu'] = [np.array([1, 1, 1])/60 for r in range(len(inst))]
inst['load'] = 0.85
inst['imbalance'] = [np.array([18, 94, 172])/284 for r in range(len(inst))]

env = Env(J=inst['J'][0], S=inst['S'][0], D=inst['D'][0],
          gamma=inst['gamma'][0],
          t=inst['t'][0], c=inst['c'][0], r=inst['r'][0], mu=inst['mu'][0],
          load=inst['load'][0], imbalance=inst['imbalance'][0], sim=True)

inst['lab'] = [env.lab for r in range(len(inst))]
inst['method'] = methods
inst = inst[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file with this instance already exists, name: ',
          FILEPATH_INSTANCE)
else:
    inst.to_csv(FILEPATH_INSTANCE)
