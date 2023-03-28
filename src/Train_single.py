"""
Class simulates the environment one time and solves it.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from numpy import array, around
import pandas as pd
from src.Plotting import plot_pi, plot_v
from src.Env import TimeConstraintEDs as Env
from src.Learner import PolicyIteration, ValueIteration, \
    OneStepPolicyImprovement

np.set_printoptions(precision=4, linewidth=150, suppress=True)
# pd.options.display.float_format = '{:.4f}'.format
np.random.seed(42)

# ---- Problem ---- #
env = Env(J=2, S=3, load=0.75, gamma=5., D=10, P=1e3, e=1e-5, trace=True,
          convergence_check=10, print_modulo=100)

# ------ Policy Iteration ------ #
pi_learner = PolicyIteration()
pi_learner.policy_iteration(env)

# ------ Value Iteration ------ #
vi_learner = ValueIteration(env, pi_learner)
vi_learner.value_iteration(env)
vi_learner.get_policy(env)

# ------ One Step Policy Improvement ------ #
ospi_learner = OneStepPolicyImprovement(env, pi_learner)
ospi_learner.get_g(env)

learner = pi_learner
def summarize_policy(env, learner):
    unique, counts = np.unique(learner.Pi, return_counts=True) 
    counts = counts/sum(counts)
    policy = pd.DataFrame(np.asarray((unique, counts)).T, columns=['Policy',
                                                                   'Freq'])
    policy.Policy = ['Keep Idle' if x == env.KEEP_IDLE
                     else 'None Waiting' if x == env.NONE_WAITING
                     else 'Servers Full' if x == env.SERVERS_FULL
                     else 'Invalid State' if x == env.NOT_EVALUATED
                     else 'Class ' + str(x) for x in policy.Policy]
    print(learner.name, 'g=', around(learner.g, int(-np.log10(env.e)-1)))   
    print(policy)
    
    feature_list = ['Class_'+str(i+1) for i in range(env.J)]
    feature_list.extend(['Keep_Idle', 'None_Waiting', 'Servers_Full',
                         'Invalid_State'])
    counts = pd.DataFrame(0, index=np.arange(env.D), columns=feature_list)
    for x in range(env.D):
        states = [slice(None)] * (1 + env.J*2)
        states[1] = x
        counts.loc[x, 'None_Waiting'] = np.sum(learner.Pi[tuple(states)]
                                               == env.NONE_WAITING)
        counts.loc[x, 'Servers_Full'] = np.sum(learner.Pi[tuple(states)]
                                               == env.SERVERS_FULL)
        counts.loc[x, 'Invalid_State'] = np.sum(learner.Pi[tuple(states)]
                                                == env.NOT_EVALUATED)
        total = np.prod(learner.Pi[tuple(states)].shape)
        total_valid = (total - counts.None_Waiting[x] - counts.Servers_Full[x]
                       - counts.Invalid_State[x])
        total_valid = 1 if total_valid == 0 else total_valid
        for i in range(env.J):
            states = [slice(None)]*(env.J*2)
            states[1+i] = x
            counts.loc[x, 'Class_'+str(i+1)] = np.sum(learner.Pi[tuple(states)]
                                                      == i+1) / total_valid
        counts.loc[x, 'Keep_Idle'] = (np.sum(learner.Pi[tuple(states)]
                                             == env.KEEP_IDLE)
                                      / total_valid)
    counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] = \
        counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] / total
    print(counts)


summarize_policy(env, pi_learner)
summarize_policy(env, vi_learner)
summarize_policy(env, ospi_learner)

# x = 0
# states = [slice(None)]*(env.J*2)
# states[0] = x
# np.sum(pi_learner.Pi[tuple(states)] == pi_learner.SERVERS_FULL)
# pi_learner.Pi[tuple(states)]
# VI_learner.Pi[tuple(states)]
# ospi_learner.Pi[tuple(states)]

print('g of OSPI:', ospi_learner.g)
print('g of VI:', vi_learner.g)
print('g of PI:', pi_learner.g)
print('Optimality gap (VI)', around(abs(ospi_learner.g-vi_learner.g)
                                    / vi_learner.g,
                                    int(-np.log10(env.e)-1))*100, '%')
print('Optimality gap (PI)', around(abs(ospi_learner.g-pi_learner.g)
                                    / pi_learner.g,
                                    int(-np.log10(env.e)-1))*100, '%')

plot_learner = 'ospi'  # pi, vi, ospi
if plot_learner == 'pi':
    Pi = pi_learner.Pi
    V = pi_learner.V
elif plot_learner == 'vi':
    Pi = vi_learner.Pi
    V = vi_learner.V
elif plot_learner == 'ospi':
    Pi = ospi_learner.Pi
    V = ospi_learner.V

if env.J > 1:
    plot_pi(env, env, Pi, zero_state=True)
    plot_pi(env, env, Pi, zero_state=False)
for i in range(env.J):
    plot_pi(env, env, Pi, zero_state=True, i=i)
    plot_pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_v(env, V, zero_state=True, i=i)
