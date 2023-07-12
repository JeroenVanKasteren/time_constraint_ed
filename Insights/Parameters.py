"""
Creation of file with instances of the Time ConstraintEDs problem.
ID of instance is the ID of the instance file combined with the index of the
instance in the file.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 31-5-2023.
"""

import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
from Env_and_Learners import TimeConstraintEDs as Env
from sklearn.model_selection import ParameterGrid

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
# TODO how to reach other folder with path!?
instances_path = 'results/instances_01.csv'
instance_columns = ['J', 'S', 'D', 'size', 'size_i',
                    'gamma', 'e', 't', 'c', 'r',
                    'lambda', 'mu', 'load', 'target_prob',
                    'vi_attempts', 'vi_time', 'vi_iter', 'vi_solved',
                    'ospi_attempts', 'ospi_time', 'ospi_iter', 'ospi_solved']

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
grid['mu'] = [[0]*J]*len(grid)
grid['lambda'] = [[0]*J]*len(grid)
grid['t'] = [[0]*J]*len(grid)
grid['c'] = [[0]*J]*len(grid)
grid['r'] = [[0]*J]*len(grid)
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
        grid['mu'].iloc[i][j] = env.mu[j]
        grid['lambda'].iloc[i][j] = env.lab[j]
        grid['t'].iloc[i][j] = env.t[j]
        grid['c'].iloc[i][j] = env.c[j]
        grid['r'].iloc[i][j] = env.r[j]
print(grid[grid['target_prob'] > MAX_TARGET_PROB])
grid = grid[grid['target_prob'] > MAX_TARGET_PROB]

# Derive solved from value for g.
grid['vi_attempts'] = 0
grid['vi_time'] = np.nan
grid['vi_iter'] = np.nan
grid['vi_solved'] = False
grid['ospi_attempts'] = 0
grid['ospi_time'] = np.nan
grid['ospi_iter'] = np.nan
grid['ospi_solved'] = False

grid = grid[instance_columns]

# TODO how to reach other folder with path!?
os.listdir()
if os.path.isfile(instances_path):
    print('file already exists, name: ', instances_path)
else:
    grid.to_csv(instances_path)

# If results file already exists
#
# write
# with open('dict.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in mydict.items():
#        writer.writerow([key, value])
#
# # read back
# with open('dict.csv') as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = dict(reader)

# def load_args(raw_args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--J', default=2)  # User input
#     parser.add_argument('--gamma', default=10)  # User input
#     args = parser.parse_args(raw_args)
#     args.id = int(args.id)
#     args.index = int(args.index)
#     args.J = int(args.J)
#     args.gamma = float(args.gamma)
#     args.policy = args.policy == 'True'
#     return args
#
# def main(raw_args=None):
#     Path(FILEPATH).touch()
#     parameters = pd.read_csv(FILEPATH, names=COLUMNS)
#     if os.stat(FILEPATH).st_size == 0:
#         with open(FILEPATH, 'r') as f:
#
#     args = load_args(raw_args)
#     # ---- Problem ---- #
#     seed = args.id * args.index
#     f_name = 'Results/' + str(args.id) + '_' + str(args.index) + 'Py.txt'
#     rho = 1
#     smu = 0
#     while smu * (1 - rho) < -np.log(MAX_TARGET_PROB):
#         env = Env(J=args.J, gamma=args.gamma, P=1e3, e=5e-4, seed=seed,
#                   max_time=args.time, convergence_check=20, print_modulo=100,
#                   b_out_f=args.b_out_f, out_f=f_name)
#         smu = env.S * sum(env.lab) / sum(env.lab / env.mu)
#         rho = env.load
#         seed += 1
#     pi_learner = PolicyIteration()