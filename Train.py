"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

python Train.py --id=1 --index=1 --J=2 --gamma=25 --policy=False --time=0-00:03:00

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import argparse
import numpy as np
from datetime import datetime
from time import perf_counter as clock
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement
from pathlib import Path

FILEPATH = 'Results/results.csv'
MAX_TARGET_PROB = 0.9

def load_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='27_1')  # SULRM_JOBID
    parser.add_argument('--index', default='27_1')  # SLURM_ARRAY_TASK_ID
    parser.add_argument('--J', default=1)  # User input
    parser.add_argument('--gamma', default=25)  # User input
    parser.add_argument('--policy', default=False)  # User input
    parser.add_argument('--time', default='0-00:02:00')  # User input
    parser.add_argument('--b_out_f', default=True)  # User input
    args = parser.parse_args(raw_args)
    args.id = int(args.id)
    args.index = int(args.index)
    args.J = int(args.J)
    args.gamma = float(args.gamma)
    args.policy = args.policy == 'True'
    return args

def main(raw_args=None):
    args = load_args(raw_args)
    # ---- Problem ---- #
    seed = args.id * args.index
    f_name = 'Results/' + str(args.id) + '_' + str(args.index) + 'Py.txt'
    rho = 1
    smu = 0
    while smu * (1 - rho) < -np.log(MAX_TARGET_PROB):
        env = Env(J=args.J, gamma=args.gamma, P=1e3, e=1e-5, seed=seed,
                  max_time=args.time, convergence_check=10, print_modulo=100,
                  b_out_f=args.b_out_f, out_f=f_name)
        smu = env.S * sum(env.lab) / sum(env.lab / env.mu)
        rho = env.load
        seed += 1
    pi_learner = PolicyIteration()

    # ---- Value Iteration ---- #
    vi_learner = ValueIteration(env, pi_learner)
    vi_learner.value_iteration(env)

    # ---- One Step Policy Improvement ---- #
    ospi_learner = OneStepPolicyImprovement(env, pi_learner)
    ospi_learner.get_g(env)

    result = [args.id, args.index, datetime.today().strftime('%Y-%m-%d'),
              seed, env.J, env.S, env.D, env.size, env.size_i, env.gamma, env.e,
              env.t, env.c, env.r, env.lab, env.mu, env.load, env.cap_prob,
              env.weighted_cap_prob, vi_learner.converged,
              ospi_learner.converged, (clock() - env.start_time),
              vi_learner.g, ospi_learner.g,
              abs(vi_learner.g - ospi_learner.g) / vi_learner.g]

    if args.policy:
        vi_learner.get_policy(env)
        np.savez('Results/policy_' + args.id + '.npz',
                 vi_learner.Pi, ospi_learner.Pi,
                 vi_learner.V, ospi_learner.V_app)
        # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')

    Path(FILEPATH).touch()
    with open(FILEPATH, 'a') as f:  # a = append
        f.write(','.join(map(str, result)) + '\n')


if __name__ == '__main__':
    main()
