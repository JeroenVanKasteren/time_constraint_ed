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
from utils import instances_sim
import pandas as pd

ID = 'sim'  # 'J2', 'J3', 'J2_D_gam', 'J1_D', 'sim'
FILEPATH_INSTANCE = 'results/instances_' + ID + '.csv'
solve = False  # False for sim, True for solve
sim_ids = range(1, 11 + 1)  # only for ID = 'sim'
max_target_prob = 0.9
remove_max_t_prob = True
max_size = 2e6
remove_max_size = True

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
    instance_columns.extend(['N', 'start_K', 'batch_T'])
    method_columns = ['_job_id', '_attempts', '_time', '_iter',
                      '_g', '_g_ci', '_perc', '_perc_ci']
    heuristic_columns = []

for method in methods:
    instance_columns.extend([method + s for s in method_columns])
    if solve and method != 'vi':
        instance_columns.extend([method + s for s in heuristic_columns])

# state space (J * D^J * S^J)
# J = 1; D = 25*20; S = 5  # D = gamma * gamma_multi
# print(f'{(J + 1) * D**J * S**J / 1e6} x e6')
if ID == 'J2':
    mu = 1 / 3
    param_grid = {'J': [2],
                  'S': [2, 5],
                  'D': [0],  # D=y*y_multi if < 0, formula if 0, value if > 0
                  'gamma': [15],
                  'mu': [[mu, mu], [mu, 1.5*mu], [mu, 2*mu]],
                  'load': [0.6, 0.8, 0.9],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'J3':  # Instance 3
    mu = 1 / 3
    param_grid = {'J': [3],
                  'S': [2, 3],
                  'D': [0, -4],
                  'gamma': [10],
                  'mu': [[mu, 1.5*mu, 2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1/3, 2/3, 1], [1, 1, 1]]}
elif ID == 'J2_D_gam':  # Instance 2  # gamma multi = 8
    mu = 1 / 3
    param_grid = {'J': [2],
                  'S': [2, 3],
                  'D': [0, -5, -10, -15],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'J1_D':
    mu = 1 / 3
    param_grid = {'J': [1],
                  'S': [2, 5],
                  'D': [0, -5, -10, -15, -20],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu], [2*mu]],
                  'load': [0.7, 0.9],
                  'imbalance': [1]}
elif ID == 'sim':
    grid = pd.DataFrame()
    for sim_id in sim_ids:
        param_grid = instances_sim.generate_instance(sim_id)
        for key, value in param_grid.items():
            param_grid[key] = [value]  # put in lists for ParameterGrid
        row = tools.get_instance_grid(param_grid, sim=True)
        grid = pd.concat([grid, row], ignore_index=True)
elif ID == 'plot_J1':  # Boxplot of D/gamma for different gamma and J
    mu = 1 / 3
    param_grid = {'J': [2],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2 * mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1 / 3, 1], [1, 1], [3, 1]]}
elif ID == 'plot_J2':
    mu = 1 / 3
    param_grid = {'J': [2],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu], [mu, 2*mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1/3, 1], [1, 1], [3, 1]]}
elif ID == 'plot_J3':
    param_grid = {'J': [3],
                  'S': list(range(2, 10)),
                  'D': [0],
                  'gamma': [10, 15, 20, 25],
                  'mu': [[mu, mu, mu], [mu, 1.5*mu, 2*mu]],
                  'load': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'imbalance': [[1/3, 2/3, 1], [1, 1, 1], [1, 2/3, 1/3]]}
    grid = tools.get_instance_grid(param_grid)
else:
    print('Error: ID not recognized')
    exit(0)

if ID != 'sim':
    grid = tools.get_instance_grid(param_grid)

print('Instances where target_prob > ', max_target_prob, ':',
      grid[grid['target_prob'] > max_target_prob])
if remove_max_t_prob:
    grid = grid[grid['target_prob'] < max_target_prob]
    print('removed')
print('Instances where size > ', max_size, ':',
      grid[grid['target_prob'] > max_size])
if remove_max_size:
    grid = grid[grid['size'] < max_size]
    print('removed')

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
