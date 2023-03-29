"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from Env import TimeConstraintEDs as Env
from Learner import PolicyIteration, ValueIteration, OneStepPolicyImprovement

np.set_printoptions(precision=4, linewidth=150, suppress=True)
np.random.seed(42)
EXPERIMENTS = 10

pi_learner = PolicyIteration()

# ---- Problem ---- #
env = Env(J=1, lmbda=array([1]), t=array([1]),
          r=array([1]), c=array([1]),
          gamma=5, D=20)

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
file_name = 'J_' + env.J + '_S_' + env.S + '_D_' + env.D + 'result.txt'
result = [env.J, env.S, env.D, env.gamma, env.e,
            env.t, env.c, env.r, env.lab, env.mu, env.Rho, env.cap_prob,
            vi_learner.g, ospi_learner.g, abs(vi_learner.g-ospi_learner.g)/vi_learner.g]
with open(file_name, 'w') as f:
    f.write(str([result]) + '\n')
