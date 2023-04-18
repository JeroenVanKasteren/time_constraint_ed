"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import argparse
import numpy as np
from datetime import datetime
from src.Env import TimeConstraintEDs as Env
from src.Learner import PolicyIteration, ValueIteration, \
    OneStepPolicyImprovement
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument('- -id', default='27')  # SLURM_ARRAY_TASK_ID
parser.add_argument('- -multiplier', default=42)  # User input
parser.add_argument('- -J', default=2)  # User input
parser.add_argument("- -gamma", default=30)  # User input
parser.add_argument("- -policy", default=False)  # User input
args = parser.parse_args()

filepath = 'Results/results.csv'

seed_id = re.sub("[^0-9]", "", args.id)
seed = seed_id * args.random_multiplier

pi_learner = PolicyIteration()

# ---- Problem ---- #
env = Env(J=args.J, S=1, load=0.75, gamma=args.gamma, D=100, P=1e3, e=1e-5,
          trace=True, convergence_check=10, print_modulo=100, seed=seed)

# ---- Value Iteration ---- #
vi_learner = ValueIteration(env, pi_learner)
vi_learner.value_iteration(env)

# ---- One Step Policy Improvement ---- #
ospi_learner = OneStepPolicyImprovement(env, pi_learner)
ospi_learner.get_g(env)

result = [datetime.today().strftime('%Y-%m-%d'),
          str([args.random_seed, args.random_multiplier]),
          env.J, env.S, env.D, env.gamma, env.e,
          env.t, env.c, env.r, env.lab, env.mu, env.load, env.cap_prob,
          vi_learner.g, ospi_learner.g,
          abs(vi_learner.g-ospi_learner.g)/vi_learner.g]

if args.policy:
    vi_learner.get_policy(env)
    np.savez('Results/policy_' + id + '.npz',
             vi_learner.Pi, ospi_learner.Pi,
             vi_learner.V, ospi_learner.V_app)
    # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')

Path(filepath).touch()
with open(filepath, 'a') as f:  # a = append
    f.write(",".join(map(str, result)) + '\n')
