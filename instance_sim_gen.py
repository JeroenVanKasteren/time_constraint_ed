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

FILEPATH_INSTANCE = 'results/instances_sim_01.csv'
input_columns = ['J', 'S', 'D', 'gamma', 't', 'c', 'r', 'lab', 'mu', 'load']
instance_columns = ['J', 'S', 'D', 'size', 'size_i',
                    'gamma', 'e', 't', 'c', 'r', 'P',
                    'lab', 'mu', 'load', 'target_prob',
                    'N', 'start_up', 'batch', 'ospi_g', 'ospi_var',
                    'fcfs_g', 'fcfs_var',
                    'sdf_g', 'sdf_var',
                    'cmu_g', 'cmu_var',
                    'sdf_prior_g', 'sdf_prior_var']

grid = pd.DataFrame(columns=input_columns)

i = 0
grid[i, 'J'] = 3
grid[i, 'S'] = 5
grid[i, 't'] = np.array([30, 60, 120])
grid[i, 'D'] = 240
grid[i, 'gamma'] = 1/5
grid[i, 'c'] = np.array([1, 1, 1])
grid[i, 'r'] = np.array([1, 1, 1])
grid[i, 'lab'] = np.array([14/60*0.1, 14/60*0.4, 14/60*0.5])
grid[i, 'mu'] = np.array([1/6.5, 1/6.5, 1/6.5])
param_grid = {'S': [2, 5, 10],
              'mu_1': MU_1_GRID,
              'mu_2': np.array([1, 1.5, 2]) * MU_1_GRID,
              'load': [0.5, 0.6, 0.7, 0.8],  # 0.9?
              'imbalance': [1 / 3, 1, 3]}

grid['target_prob'] = 0
grid['D'] = [[0] * J] * len(grid)
grid['size'] = 0
grid['size_i'] = 0
grid['mu'] = [[] for r in range(len(grid))]
grid['lab'] = [[] for r in range(len(grid))]
grid['t'] = [[] for r in range(len(grid))]
grid['c'] = [[] for r in range(len(grid))]
grid['r'] = [[] for r in range(len(grid))]

for i, inst in grid.iterrows():
    env = Env(J=J, S=inst.S, gamma=gamma, P=P, e=e, t=t, c=c, r=r,
              mu=np.array([inst.mu_1, inst.mu_2]),  # TODO: generalize
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


grid = tools.get_instance_grid(J=2,
                               gamma=20,  # What value of gamma to use?
                               t=np.array([1]*J),
                               c=np.array([1]*J),
                               r=np.array([1]*J),
                               s_grid=[2, 5, 10],
                               mu_1_grid=MU_1_GRID,
                               mu_2_grid=np.array([1, 1.5, 2]) * MU_1_GRID,
                               load_grid=[0.5, 0.6, 0.7, 0.8],  # 0.9?
                               load_imb=[1 / 3, 1, 3],
                               max_target_prob=0.9)

# Derive solved from value for g.
grid['N'] = np.nan
grid['start_up'] = np.nan
grid['batch'] = np.nan
grid['ospi_g'] = np.nan
grid['ospi_var'] = np.nan
grid['fcfs_g'] = np.nan
grid['fcfs_var'] = np.nan
grid['sdf_g'] = np.nan
grid['sdf_var'] = np.nan
grid['cmu_g'] = np.nan
grid['cmu_var'] = np.nan
grid['sdf_prior_g'] = np.nan
grid['sdf_prior_var'] = np.nan
grid = grid[instance_columns]

if os.path.isfile(FILEPATH_INSTANCE):
    print('Error: file already exists, name: ', FILEPATH_INSTANCE)
else:
    grid.to_csv(FILEPATH_INSTANCE)
