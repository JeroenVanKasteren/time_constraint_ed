"""
Class simulates the environment one time and solves it.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from utils import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement, plot_pi, plot_v, tools

np.set_printoptions(precision=4, linewidth=150, suppress=True)

MAX_TARGET_PROB = 0.9

# ---- Problem ---- #
seed = 42  # np.random.randint(0, 1e8)
rho = 1
smu = 0
while smu * (1 - rho) < -np.log(MAX_TARGET_PROB):
    env = Env(J=1, S=5, gamma=1, D=200, t=np.array([60]),
              mu=np.array([0.033333]), lab=np.array([0.1416666]),
              P=1e3, e=1e-4, seed=seed,
              max_time='0-00:10:00', convergence_check=10, print_modulo=100,
              max_iter=1e4)
    smu = env.S * sum(env.lab) / sum(env.lab / env.mu)
    rho = env.load
    seed += 1
to_plot = []  # 'VI', 'PI', 'OSPI' (what to plot)

# ------ Policy Iteration ------ #
pi_learner = PolicyIteration()
# pi_learner.policy_iteration(env)

# ------ Value Iteration ------ #
vi_learner = ValueIteration(env, pi_learner)
vi_learner.value_iteration(env)
vi_learner.get_policy(env)

# ------ One Step Policy Improvement ------ #
ospi_learner = OneStepPolicyImprovement(env, pi_learner)
ospi_learner.get_g(env, ospi_learner.V)

# summarize_policy(env, pi_learner)
tools.summarize_policy(env, vi_learner)
tools.summarize_policy(env, ospi_learner)

print('g of OSPI:', np.around(ospi_learner.g, int(-np.log10(env.e)-1)))
print('g of VI:', np.around(vi_learner.g, int(-np.log10(env.e)-1)))
# print('g of PI:', np.around(pi_learner.g, int(-np.log10(env.e)-1)))
print('Optimality gap (VI)', np.around(abs(ospi_learner.g-vi_learner.g)
                                       / vi_learner.g,
                                       int(-np.log10(env.e)-1))*100, '%')
# print('Optimality gap (PI)', np.around(abs(ospi_learner.g-pi_learner.g)
#                                        / pi_learner.g,
#                                        int(-np.log10(env.e)-1))*100, '%')

for plot_learner in to_plot:
    if plot_learner == 'VI':
        Pi = vi_learner.Pi
        V = vi_learner.V
        name = 'VI'
    elif plot_learner == 'PI':
        Pi = pi_learner.Pi
        V = pi_learner.V
        name = 'PI'
    elif plot_learner == 'OSPI':
        Pi = ospi_learner.Pi
        V = ospi_learner.V_app
        name = 'OSPI'
    if env.J > 1:
        plot_pi(env, env, Pi, zero_state=True, name=name)
        plot_pi(env, env, Pi, zero_state=False, name=name)
    for i in range(env.J):
        plot_pi(env, env, Pi, zero_state=True, i=i, name=name)
        plot_pi(env, env, Pi, zero_state=True, i=i, smu=True, name=name)
        plot_v(env, V, zero_state=True, i=i, name=name)
