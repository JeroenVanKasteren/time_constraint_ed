"""
Creation of instance file for the Time ConstraintEDs problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 31-5-2023.
"""

import numpy as np
import os
from utils import tools

FILEPATH_INSTANCE = 'results/instances_02.csv'
instance_columns = ['J', 'S', 'D', 'size', 'size_i',
                    'gamma', 'e', 't', 'c', 'r', 'P',
                    'lab', 'mu', 'load', 'target_prob']
methods = ['vi', 'ospi', 'sdf']

columns = ['_job_id', '_attempts', '_time', '_iter', '_g_tmp', '_g']
heuristic_columns = ['_opt_gap_tmp', '_opt_gap']
for method in methods:
    instance_columns.extend([method + s for s in columns])
    if method != 'vi':
        instance_columns.extend([method + s for s in heuristic_columns])

J = 2
MU_1_GRID = [1/3]

param_grid = {'S': [2, 5, 10],
              'mu_1': MU_1_GRID,
              'mu_2': np.array([1, 1.5, 2]) * MU_1_GRID,
              'load': [0.5, 0.6, 0.7, 0.8],  # 0.9?
              'imbalance': [1 / 3, 1, 3]}
# Idea for J > 2
# mu = ['mu_']*J
# for i in range(J):
#     mu[i] += str(i+1)

grid = tools.get_instance_grid(J=2,
                               gamma=15,
                               e=1e-5,
                               P=1e3,
                               t=np.array([1] * J),
                               c=np.array([1] * J),
                               r=np.array([1] * J),
                               param_grid=param_grid,
                               max_target_prob=0.9)

# Derive solved from value for g.
for method in methods:
    grid[method + '_job_id'] = ''
    grid[method + '_attempts'] = 0
    grid[method + '_time'] = '00:00'
    grid[method + '_iter'] = 0
    grid[method + '_g_tmp'] = np.nan
    grid[method + '_g'] = np.nan
    if method != 'vi':
        grid[method + '_opt_gap_tmp'] = np.nan
        grid[method + '_opt_gap'] = np.nan

grid = grid[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file already exists, name: ', FILEPATH_INSTANCE)
else:
    grid.to_csv(FILEPATH_INSTANCE)
