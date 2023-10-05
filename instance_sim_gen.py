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

FILEPATH_INSTANCE = 'results/instance_sim_01.csv'
methods = ['ospi', 'cmu', 'fcfs', 'sdf', 'sdf_prior']
input_columns = ['J', 'S', 'gamma', 'D', 't', 'c', 'r', 'mu', 'load',
                 'imbalance']
instance_columns = [*input_columns, 'N', 'start_K', 'batch_T',
                    'method', 'g', 'var']

inst = pd.DataFrame(0, index=np.arange(len(methods)), columns=instance_columns)

inst['J'] = 3
inst['S'] = 5
inst['t'] = np.array([30, 60, 120])
inst['D'] = 240
inst['gamma'] = 1/5
inst['c'] = np.array([1, 1, 1])
inst['r'] = np.array([1, 1, 1])
# inst['lab'] = np.array([14/60*0.1, 14/60*0.4, 14/60*0.5])
inst['mu'] = np.array([1/6, 1/12, 1/18])
inst['load'] = 0.75
inst['imbalance'] = np.array([0.5, 0.4, 0.1])

# grid['mu'] = [[] for r in range(len(grid))]
# grid['lab'] = [[] for r in range(len(grid))]
# grid['t'] = [[] for r in range(len(grid))]
# grid['c'] = [[] for r in range(len(grid))]
# grid['r'] = [[] for r in range(len(grid))]

env = Env(J=inst[0, 'J'], S=S, D=D, gamma=gamma, t=t, c=c, r=r, mu=mu, load=load,
          imbalance=imbalance)

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file with this instance already exists, name: ',
          FILEPATH_INSTANCE)
else:
    inst.to_csv(FILEPATH_INSTANCE)
