"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from datetime import datetime
from src.Env import TimeConstraintEDs as Env
from src.Learner import PolicyIteration, ValueIteration, OneStepPolicyImprovement
from pathlib import Path

np.set_printoptions(precision=4, linewidth=150, suppress=True)
np.random.seed(42)
filepath = 'Results/results.csv'
EXPERIMENTS = 10

pi_learner = PolicyIteration()

# ---- Problem ---- #
env = Env(J=1, S=1, load=0.75, gamma=15., D=100, P=1e3, e=1e-5, trace=True,
          convergence_check=10, print_modulo=100, max_iter=4000)

# ------ Value Iteration ------ #
vi_learner = ValueIteration(env, pi_learner)
vi_learner.value_iteration(env)
vi_learner.get_policy(env)

# ------ One Step Policy Improvement ------ #
ospi_learner = OneStepPolicyImprovement(env, pi_learner)
ospi_learner.get_g(env)

# ['J', 'S', 'D', 'gamma', 'eps',
#  't', 'c', 'r', 'lambda', 'mu', 'Rho', 'cap_prob',
#  'VI', 'OSPI', 'gap']

result = [datetime.today().strftime('%Y-%m-%d'),
          env.J, env.S, env.D, env.gamma, env.e,
          env.t, env.c, env.r, env.lab, env.mu, env.load, env.cap_prob,
          vi_learner.g, ospi_learner.g,
          abs(vi_learner.g-ospi_learner.g)/vi_learner.g]

Path(filepath).touch()
with open(filepath, 'a') as f:  # a = append
    f.write(",".join(map(str, result)) + '\n')
