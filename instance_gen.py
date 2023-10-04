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

FILEPATH_INSTANCE = 'results/sim_instances_01.csv'
instance_columns = ['J', 'S', 'D', 'gamma', 'e', 't', 'c', 'r', 'P',
                    'lab', 'mu', 'load', 'target_prob',
                    'vi_job_id', 'vi_attempts', 'vi_time', 'vi_iter',
                    'vi_g_tmp', 'vi_g',
                    'ospi_job_id',
                    'ospi_attempts', 'ospi_time', 'ospi_iter',
                    'vi_g_tmp', 'ospi_g',
                    'opt_gap']

J = 3
S = 5
gamma = 30
D = 100
t = np.array([30, 60, 120])
c = np.array([1]*J)
r = np.array([1]*J)
# lab = np.array([14/60*0.5, 14/60*0.4, 14/60*0.1])
imbalance = np.array([0.5, 0.4, 0.1])
mu = np.array([1/10, 1/20, 1/30])
load = 0.75
convergence_check = 1e4
env = Env(J=J, S=S, D=D, gamma=gamma, t=t, c=c, r=r, mu=mu, load=load,
          imbalance=imbalance)
# lab=lab, e=0.1, max_time=args.time)
lab = env.lab
p_xy = env.p_xy
regret = np.max(r) - r + c
cmu = c * mu

mu = ['mu_']*J  # TODO
for i in range(J):
    mu[i] += str(i+1)
grid = tools.generate_instance_grid(J=2,
                                    gamma=15,
                                    e=1e-5,
                                    P=1e3,
                                    t=np.array([1]*J),
                                    c=np.array([1]*J),
                                    r=np.array([1]*J),
                                    s_grid=[2, 5, 10],
                                    param_grid=param_grid,
                                    max_target_prob=0.9)
grid.loc[i, 'target_prob'] = env.target_prob
grid.loc[i, 'D'] = env.D
grid.loc[i, 'size'] = env.size
grid.loc[i, 'size_i'] = env.size_i
grid.loc[i, 'mu'].append(env.mu[j])
grid.loc[i, 'lab'].append(env.lab[j])
grid.loc[i, 't'].append(env.t[j])
grid.loc[i, 'c'].append(env.c[j])
grid.loc[i, 'r'].append(env.r[j])

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
