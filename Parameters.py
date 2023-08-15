"""
Creation of file with instances of the Time ConstraintEDs problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 31-5-2023.
"""

import numpy as np
import os
import pandas as pd

from sklearn.model_selection import ParameterGrid
from utils import TimeConstraintEDs as Env

J = 2
gamma = 15
e = 1e-5
P = 1e3
t = np.array([1, 1])
c = np.array([1, 1])
r = np.array([1, 1])

S_GRID = [2, 5, 10]
MU_1_GRID = [1/3]
MU_2_GRID = np.array([1, 1.5, 2])*MU_1_GRID
LOAD_GRID = [0.5, 0.6, 0.7, 0.8]  # 0.9?
LOAD_IMB = [1/3, 1, 3]

FILEPATH_INSTANCE = 'results/instances_01.csv'
instance_columns = ['J', 'S', 'D', 'size', 'size_i',
                    'gamma', 'e', 't', 'c', 'r', 'P',
                    'lab', 'mu', 'load', 'target_prob',
                    'vi_job_id', 'vi_attempts', 'vi_time', 'vi_iter', 'vi_g',
                    'ospi_job_id',
                    'ospi_attempts', 'ospi_time', 'ospi_iter', 'ospi_g',
                    'opt_gap']

MAX_TARGET_PROB = 0.9

param_grid = {'S': S_GRID,
              'mu_1': MU_1_GRID,
              'mu_2': MU_2_GRID,
              'load': LOAD_GRID,
              'imbalance': LOAD_IMB}
grid = pd.DataFrame(ParameterGrid(param_grid))
print("Length of grid:", len(grid))

grid['J'] = J
grid['gamma'] = gamma
grid['e'] = e
grid['P'] = P

grid['target_prob'] = 0
grid['D'] = [[0]*J]*len(grid)
grid['size'] = 0
grid['size_i'] = 0
grid['mu'] = [[] for r in range(len(grid))]
grid['lab'] = [[] for r in range(len(grid))]
grid['t'] = [[] for r in range(len(grid))]
grid['c'] = [[] for r in range(len(grid))]
grid['r'] = [[] for r in range(len(grid))]

for i, inst in grid.iterrows():
    env = Env(J=J, S=inst.S, gamma=gamma, P=P, e=e, t=t, c=c, r=r,
              mu=np.array([inst.mu_1, inst.mu_2]),
              load=inst.load,
              imbalance=np.array([inst.imbalance, 1]))
    grid.loc[i, 'target_prob'] = env.target_prob
    grid.loc[i, 'D'] = env.D
    grid.loc[i, 'size'] = env.size
    grid.loc[i, 'size_i'] = env.size_i
    for j in range(J):
        grid.loc[i, 'mu'].append(env.mu[j])
        grid.loc[i, 'lab'].append(env.lab[j])
        grid.loc[i, 't'].append(env.t[j])
        grid.loc[i, 'c'].append(env.c[j])
        grid.loc[i, 'r'].append(env.r[j])
print('Removed instances due to target_prob > ', MAX_TARGET_PROB, ':',
      grid[grid['target_prob'] > MAX_TARGET_PROB])
grid = grid[grid['target_prob'] < MAX_TARGET_PROB]

# Derive solved from value for g.
grid['vi_job_id'] = ''
grid['vi_attempts'] = 0
grid['vi_time'] = np.nan
grid['vi_iter'] = np.nan
grid['vi_g'] = np.nan
grid['ospi_job_id'] = ''
grid['ospi_attempts'] = 0
grid['ospi_time'] = np.nan
grid['ospi_iter'] = np.nan
grid['ospi_g'] = np.nan
grid['opt_gap'] = np.nan
grid = grid[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file already exists, name: ', FILEPATH_INSTANCE)
else:
    grid.to_csv(FILEPATH_INSTANCE)
