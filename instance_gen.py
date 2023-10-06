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
                    'lab', 'mu', 'load', 'target_prob',
                    'vi_job_id', 'vi_attempts', 'vi_time', 'vi_iter',
                    'vi_g_tmp', 'vi_g',
                    'ospi_job_id',
                    'ospi_attempts', 'ospi_time', 'ospi_iter',
                    'vi_g_tmp', 'ospi_g',
                    'opt_gap']

J = 2
MU_1_GRID = [1/3]

param_grid = {'S': [2, 5, 10],
              'mu_1': MU_1_GRID,
              'mu_2': np.array([1, 1.5, 2]) * MU_1_GRID,
              'load': [0.5, 0.6, 0.7, 0.8],  # 0.9?
              'imbalance': [1 / 3, 1, 3]}
mu = ['mu_']*J  # TODO
for i in range(J):
    mu[i] += str(i+1)
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
grid['vi_job_id'] = ''
grid['vi_attempts'] = 0
grid['vi_time'] = '00:00'
grid['vi_iter'] = 0
grid['vi_g_tmp'] = np.nan
grid['vi_g'] = np.nan
grid['ospi_job_id'] = ''
grid['ospi_attempts'] = 0
grid['ospi_time'] = '00:00'
grid['ospi_iter'] = 0
grid['ospi_g_tmp'] = np.nan
grid['ospi_g'] = np.nan
grid['opt_gap'] = np.nan
grid = grid[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file already exists, name: ', FILEPATH_INSTANCE)
else:
    grid.to_csv(FILEPATH_INSTANCE)
