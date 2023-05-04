"""
Class simulates the environment one time and solves it.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
import pandas as pd
from Insights import plot_pi, plot_v
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement

np.set_printoptions(precision=4, linewidth=150, suppress=True)

WEIGHTED_CAP_PROB_MAX = 0.15

# ---- Problem ---- #
seed = np.random.randint(0, 1e8)
weighted_cap_prob = 1
while weighted_cap_prob > WEIGHTED_CAP_PROB_MAX:
    env = Env(J=2, gamma=30., P=1e3, e=1e-5, seed=seed,
              convergence_check=10, print_modulo=100, max_time='00:01:35')
    weighted_cap_prob = sum(env.cap_prob * env.lab) / sum(env.lab)
    seed += 1

to_plot = []  # 'VI', 'PI', 'OSPI' (what to plot)

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
    print(learner.name, 'g=', np.around(learner.g, int(-np.log10(env.e)-1)))
    print(policy.to_string(formatters={'Freq': '{:,.2%}'.format}))
    
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
    print(counts.to_string(formatters={'Class_1': '{:,.2%}'.format,
                                       'Class_2': '{:,.2%}'.format,
                                       'Keep_Idle': '{:,.2%}'.format,
                                       'None_Waiting': '{:,.2%}'.format,
                                       'Servers_Full': '{:,.2%}'.format,
                                       'Invalid_State': '{:,.2%}'.format}))


summarize_policy(env, pi_learner)
summarize_policy(env, vi_learner)
summarize_policy(env, ospi_learner)

print('g of OSPI:', np.around(ospi_learner.g, int(-np.log10(env.e)-1)))
print('g of VI:', np.around(vi_learner.g, int(-np.log10(env.e)-1)))
print('g of PI:', np.around(pi_learner.g, int(-np.log10(env.e)-1)))
print('Optimality gap (VI)', np.around(abs(ospi_learner.g-vi_learner.g)
                                    / vi_learner.g,
                                    int(-np.log10(env.e)-1))*100, '%')
print('Optimality gap (PI)', np.around(abs(ospi_learner.g-pi_learner.g)
                                    / pi_learner.g,
                                    int(-np.log10(env.e)-1))*100, '%')

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
