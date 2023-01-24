"""
Class simulates the environment multiple times and solves it.

Setup:
N classes, N=2,3,4 (SIMS=100 experiments each)

Server_allocation approx. Methods
0 --> integer servers, V_app(x, s) = sum V_{s*}(x)
1 --> continuous servers, V_app(x, s) = sum V_{s*}(x)
2 --> continuous servers, V_app(x, s) = sum V_{s*}(s_i - s_i* + x)
3 --> continuous servers, V_app(x, s) = sum V_{s}(s_i - s_i* + x)

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
import os

os.chdir(r"C:\Users\Desktop\surfdrive\VU\Promovendus"
         r"\Time constraints in emergency departments\Code")
# os.chdir(r"C:\Users\jkn354\Documents\surfdrive\VU\Promovendus"
#          r"\Time constraints in emergency departments\Code")

from Env import TimeConstraintEDs
from Learner import OneStepPolicyImprovement, PolicyIteration, ValueIteration
# from Tools import feasibility

np.set_printoptions(precision=5)
MAX_ITER = 10
np.random.seed(42)


# ---- Problem ---- #
env = TimeConstraintEDs(J=1, S=1,
                        mu=[0.5], lambda_=[1], t=[0],  # t=np.array([0])
                        gamma=1, D=3, P=1e2, e=1e-5,
                        max_iter=MAX_ITER,
                        trace=True, print_modulo=5, time_check=True)

# ------ Value Iteration ------ #
VI_learner = ValueIteration(env, MAX_ITER)
VI_learner.value_iteration()
print(env.print_g(VI_learner.V))
print(VI_learner.V)
print(env.tau)

# ------ Policy Iteration ------ #
PI_learner = PolicyIteration(env, MAX_ITER)
PI_learner.policy_iteration()
print(env.print_g(VI_learner.V))

# ------ One Step Policy Improvement ------ #
VI_learner = ValueIteration(env, MAX_ITER)
VI_learner.value_iteration()
print(env.print_g(VI_learner.V))
print(VI_learner.V)
print(env.tau)
