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

S_GRID = [2, 5, 10]
MU_1_GRID = [1/4]
MU_2_GRID = np.array([1, 1.5, 2])*MU_1_GRID
RHO_GRID = [0.5, 0.6, 0.7, 0.8]  # 0.9?
RHO_IMB = [1/3, 1, 3]

FILEPATH = 'Results/instances.csv'
COLUMNS = ['instance', 'job_id', 'array_id', 'date',
           'J', 'S', 'D', 'size', 'size_i',
           'gamma', 'eps', 't', 'c', 'r',
           'lambda', 'mu', 'load', 'cap_prob',
           'vi_touched', 'vi_job_id', 'vi_converged', 'vi_g',
           'vi_time', 'vi_iter', 'vi_cores',
           'ospi_touched', 'ospi_job_id', 'ospi_converged', 'ospi_g',
           'ospi_time', 'ospi_iter', 'ospi_cores',
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