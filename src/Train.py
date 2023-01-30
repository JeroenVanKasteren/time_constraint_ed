"""
Class simulates the environment multiple times and solves it.

Setup:
N classes, N=2,3,4 (SIMS=100 experiments each)

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from numpy import arange, around, array
import matplotlib.pyplot as plt

from Env import TimeConstraintEDs as Env
from Learner import PolicyIteration, ValueIteration, OneStepPolicyImprovement

np.set_printoptions(precision=4, linewidth=150, suppress=True)
np.random.seed(42)
EXPERIMENTS = 10

opt_gap_VI = []
opt_gap_PI = []

for n in arange(EXPERIMENTS):
    # ---- Problem ---- #
    env = Env(J=1, lmbda=array([1]), t=array([1]),
              r=array([1]), c=array([1]),
              gamma=5, D=20)
    
    # ------ Policy Iteration ------ #
    PI_learner = PolicyIteration(env)
    # PI_learner.policy_iteration(env)

    # ------ Value Iteration ------ #
    VI_learner = ValueIteration(env, PI_learner)
    VI_learner.value_iteration(env, PI_learner)
    opt_gap_VI.append(VI_learner.g)  # TODO
    # ------ One Step Policy Improvement ------ #
    # OSPI_learner = OneStepPolicyImprovement(env, PI_learner)
    # OSPI_learner.one_step_policy_improvement(env, PI_learner)

    # OSPI_learner.calculate_g(env, PI_learner)

    # opt_gap_VI.append(around(abs(OSPI_learner.g-VI_learner.g)/VI_learner.g, int(-np.log10(env.e)-1))*100)
    # opt_gap_PI.append(around(abs(OSPI_learner.g-PI_learner.g)/PI_learner.g, int(-np.log10(env.e)-1))*100)

print(opt_gap_VI)  # TODO
# print("g of VI & PI", np.array([opt_gap_VI, opt_gap_PI]))
# print("g of VI equal to PI?", opt_gap_VI == opt_gap_PI)
# plt.boxplot(opt_gap_VI)
# plt.title("Optimality Gap (VI)")
# plt.boxplot(opt_gap_PI)
# plt.title("Optimality Gap (PI)")
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html