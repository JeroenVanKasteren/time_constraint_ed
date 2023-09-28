"""
Creation of instance file to simulate the TimeConstraintEDs problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 31-5-2023.
"""

import numpy as np
import os
from utils import tools

FILEPATH_INSTANCE = 'results/instances_sim_01.csv'
instance_columns = ['J', 'S', 'D', 'size', 'size_i',
                    'gamma', 'e', 't', 'c', 'r', 'P',
                    'lab', 'mu', 'load', 'target_prob',
                    'N', 'start_up', 'batch', 'ospi_g', 'ospi_var',
                    'fcfs_g', 'fcfs_var',
                    'sdf_g', 'sdf_var',
                    'cmu_g', 'cmu_var',
                    'sdf_prior_g', 'sdf_prior_var']
J = 2
MU_1_GRID = [1/3]

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
