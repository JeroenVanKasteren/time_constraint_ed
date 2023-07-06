"""
Creation of file with instances of the Time ConstraintEDs problem.

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

param_grid = {'S': S_GRID,
              'mu_1': MU_1_GRID,
              'mu_2': MU_2_GRID,
              'load': LOAD_GRID,
              'imbalance': RHO_IMB}
grid = pd.DataFrame(ParameterGrid(param_grid))
print("Length of grid:", len(grid))

for inst in grid:
    env = Env(J=J, S=inst.S, gamma=gamma, P=P, e=e, t=t, c=c, r=r,
              mu=np.array([inst.mu_1, inst.mu_2]),
              load=inst.load,
              imbalance=np.array([inst.imbalance, 1]))
grid['mu'] = [[a, b] for a, b in zip(grid.mu_1, grid.mu_2)]
grid['lab'] = [[a, b] for a, b in zip(lab_1, lab_2)]
MAX_TARGET_PROB = 0.9
smu = np.array([S * (lab[0] + lab[1]) / lab[0]/mu[0] + lab[1]/mu[1]
                for S, lab, mu in zip(grid.S, grid.lab, grid.mu)])
print("# instance with too heavy load?",
      sum(smu * (1 - grid.rho) < -np.log(MAX_TARGET_PROB)))

while :
    env = Env(J=args.J, gamma=args.gamma, P=1e3, e=5e-4, seed=seed,
              max_time=args.time, convergence_check=20, print_modulo=100,
              b_out_f=args.b_out_f, out_f=f_name)

    rho = env.load
    seed += 1

# # write
# with open('dict.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in mydict.items():
#        writer.writerow([key, value])
#
# # read back
# with open('dict.csv') as csv_file:
#     reader = csv.reader(csv_file)
#     mydict = dict(reader)

FILEPATH = 'Results/instances.csv'
COLUMNS = ['instance',
           'job_id_vi', 'array_id_vi', 'solve_date_vi',
           'job_id_ospi', 'array_id_ospi', 'solve_date_ospi',
           'J', 'S', 'D', 'size', 'size_i',
           'gamma', 'eps', 't', 'c', 'r',
           'lambda', 'mu', 'load',
           'vi_g', 'vi_time', 'vi_iter',
           'ospi_g', 'ospi_time', 'ospi_iter',
           'rel_gap', 'abs_gap']

MAX_TARGET_PROB = 0.9

def load_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--J', default=2)  # User input
    parser.add_argument('--gamma', default=10)  # User input
    args = parser.parse_args(raw_args)
    args.id = int(args.id)
    args.index = int(args.index)
    args.J = int(args.J)
    args.gamma = float(args.gamma)
    args.policy = args.policy == 'True'
    return args

def main(raw_args=None):
    Path(FILEPATH).touch()
    parameters = pd.read_csv(FILEPATH, names=COLUMNS)
    if os.stat(FILEPATH).st_size == 0:
        with open(FILEPATH, 'r') as f:

    args = load_args(raw_args)
    # ---- Problem ---- #
    seed = args.id * args.index
    f_name = 'Results/' + str(args.id) + '_' + str(args.index) + 'Py.txt'
    rho = 1
    smu = 0
    while smu * (1 - rho) < -np.log(MAX_TARGET_PROB):
        env = Env(J=args.J, gamma=args.gamma, P=1e3, e=5e-4, seed=seed,
                  max_time=args.time, convergence_check=20, print_modulo=100,
                  b_out_f=args.b_out_f, out_f=f_name)
        smu = env.S * sum(env.lab) / sum(env.lab / env.mu)
        rho = env.load
        seed += 1
    pi_learner = PolicyIteration()