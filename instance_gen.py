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

solve = True  # False for sim, True for solve
ID = 'J2'
FILEPATH_INSTANCE = 'results/instances_' + ID + '.csv'

if not solve:
    FILEPATH_INSTANCE += '_sim'

instance_columns = ['J', 'S', 'gamma', 'D',
                    'mu', 'lab', 'load', 'imbalance'
                    't', 'c', 'r',
                    'max_t_prob']
if solve:
    methods = ['vi', 'ospi', 'sdf', 'fcfs', 'pi']
    instance_columns.extend(['e', 'P', 'size', 'size_i'])
    method_columns = ['_job_id', '_attempts', '_time', '_iter', '_g_tmp', '_g']
    heuristic_columns = ['_opt_gap_tmp', '_opt_gap']
else:
    methods = ['ospi', 'cmu_t_min', 'cmu_t_max', 'fcfs', 'sdf', 'sdfprior',
               'l_max', 'l_min']
    instance_columns.append(['N', 'start_K', 'batch_T'])
    method_columns = ['_job_id', '_attempts', '_time', '_iter',
                      '_g', '_g_ci', '_perc', '_perc_ci']

for method in methods:
    instance_columns.extend([method + s for s in method_columns])
    if solve and method != 'vi':
        instance_columns.extend([method + s for s in heuristic_columns])

mu = 1/3
if ID == 'J2':
    J = 2
    param_grid = {'S': [2, 5],
                  'D': [0],  # =gamma_multi if > 0
                  'gamma': [15],
                  'mu': [[mu, mu], [mu, 1.5*mu], [mu, 2*mu]],
                  'load': [0.6, 0.8, 0.9],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'J3_D4':  # Instance 3
    J = 3
    param_grid = {'S': [2, 3],
                  'D': [4],
                  'gamma': [10],
                  'mu': [[mu, 1.5*mu, 2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1/3, 2/3, 1], [1, 1, 1]]}
elif ID == 'J3':
    J = 3
    param_grid = {'S': [2, 3],
                  'D': [0],
                  'gamma': [10],
                  'mu': [[mu, 1.5 * mu, 2 * mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1 / 3, 2 / 3, 1], [1, 1, 1]]}
elif ID == 'J2_gam':  # Instance 2  # gamma multi = 8
    J = 2
    param_grid = {'S': [2, 4],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'J1_D':
    J = 1
    param_grid = {'S': [2, 4],
                  'D': [0, 5, 10],
                  'gamma': [15],
                  'mu': [[mu], [2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1]]}

# state space (J * D^J * S^J)
# (2 + 1) * (20*8)**2 * 5**2  # D = gamma * gamma_multi
# (3 + 1) * (10*4)**3 * 2**3

grid = tools.get_instance_grid(J=J,
                               gamma_multi=gamma_multi,  # 0 for D-formula
                               e=1e-4,
                               P=1e3,
                               t=np.array([1] * J),
                               c=np.array([1] * J),
                               r=np.array([1] * J),
                               param_grid=param_grid,
                               max_target_prob=0.9)

# Derive solved from g value.
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
