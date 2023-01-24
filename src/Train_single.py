"""
Class simulates the environment one time and solves it.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from numpy import array, arange, around
import pandas as pd
import os

# PATH = (r"D:\Programs\Surfdrive\Surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")
PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")
os.chdir(PATH)
from Plotting import plot_Pi, plot_V

from Env import TimeConstraintEDs as Env
from Learner import PolicyIteration, ValueIteration, OneStepPolicyImprovement
# np.set_printoptions(precision=4)
# pd.options.display.float_format = '{:.4f}'.format
np.random.seed(42)

# ---- Problem ---- #
# env = Env(J=1, S=3, mu=array([1]), lmbda=array([1.5]), t=array([3]),
#           r=array([1]), c=array([1]), P=0, e=1e-6,
#           gamma=2, D=8, trace=False)
# env = Env(J=2, S=4, mu=array([0.5,1.5]), lmbda=array([1,1]), t=array([3,3]),
#           r=array([1,1]), c=array([1,1]), P=0, e=1e-5, 
#           gamma=2, D=10)
# env = Env(J=2, S=3, mu=array([1,1]), lmbda=array([1,1]), t=array([3,3]),
#           r=array([1,1]), c=array([1,1]), P=0, e=1e-5,
#           gamma=2, D=14, time_check=True)
env = Env(J=2, S=5, mu=array([0.9447,0.1007]), lmbda=array([2.3586,0.2497]), t=array([1,1]),
          r=array([1,1]), c=array([1,1]), P=0, e=1e-5,gamma=5, D=10)
env.tau
# ------ Policy Iteration ------ #
PI_learner = PolicyIteration(env)
PI_learner.policy_iteration(env)

# ------ Value Iteration ------ #
VI_learner = ValueIteration(env, PI_learner)
VI_learner.value_iteration(env, PI_learner)

# ------ One Step Policy Improvement ------ #
OSPI_learner = OneStepPolicyImprovement(env, PI_learner)
OSPI_learner.one_step_policy_improvement(env, PI_learner)
OSPI_learner.calculate_g(env, PI_learner)
PI_learner2 = PolicyIteration(env)
PI_learner2.Pi = OSPI_learner.Pi.copy()
PI_learner2.policy_iteration(env)

def summarize_policy(env, learner, PI_learner):
    unique, counts = np.unique(learner.Pi, return_counts=True) 
    counts = counts/sum(counts)
    policy = pd.DataFrame(np.asarray((unique, counts)).T, columns=['Policy', 'Freq'])
    policy.Policy = ['Keep Idle' if x == PI_learner.KEEP_IDLE
                   else 'None Waiting' if x == PI_learner.NONE_WAITING
                   else 'Servers Full' if x == PI_learner.SERVERS_FULL
                   else 'Invalid State' if x == PI_learner.NOT_EVALUATED
                   else 'Class '+str(x) for x in policy.Policy]
    print(learner.name, 'g=', around(learner.g, int(-np.log10(env.e)-1)))   
    print(policy)
    
    feature_list = ['Class_'+str(i+1) for i in arange(env.J)]
    feature_list.extend(['Keep_Idle', 'None_Waiting', 'Servers_Full', 'Invalid_State'])
    counts = pd.DataFrame(0, index=np.arange(env.D), columns=feature_list)
    for x in arange(env.D):
        states = [slice(None)]*(env.J*2)
        states[0] = x
        counts.loc[x, 'None_Waiting'] = np.sum(learner.Pi[tuple(states)] == PI_learner.NONE_WAITING)
        counts.loc[x, 'Servers_Full'] = np.sum(learner.Pi[tuple(states)] == PI_learner.SERVERS_FULL)
        counts.loc[x, 'Invalid_State'] = np.sum(learner.Pi[tuple(states)] == PI_learner.NOT_EVALUATED)
        total = np.prod(learner.Pi[tuple(states)].shape)
        total_valid = total - counts.None_Waiting[x] - counts.Servers_Full[x] - counts.Invalid_State[x]
        total_valid = 1 if total_valid == 0 else total_valid
        for i in arange(env.J):
            states = [slice(None)]*(env.J*2)
            states[i] = x
            counts.loc[x, 'Class_'+str(i+1)] = np.sum(learner.Pi[tuple(states)] == i+1) / total_valid
        counts.loc[x, 'Keep_Idle'] = np.sum(learner.Pi[tuple(states)] == PI_learner.KEEP_IDLE) / total_valid
    counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] = \
        counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] / total
    print(counts)

summarize_policy(env, PI_learner, PI_learner)
summarize_policy(env, VI_learner, PI_learner)
summarize_policy(env, OSPI_learner, PI_learner)

PI_learner.Pi
x = 0
states = [slice(None)]*(env.J*2)
states[0] = x
np.sum(PI_learner.Pi[tuple(states)] == PI_learner.SERVERS_FULL)
PI_learner.Pi[tuple(states)]
VI_learner.Pi[tuple(states)]
OSPI_learner.Pi[tuple(states)]

print('g of OSPI:', OSPI_learner.g)
print('g of VI:', VI_learner.g)
print('g of PI:', PI_learner.g)
print('Optimality gap (VI)', around(abs(OSPI_learner.g-VI_learner.g)/VI_learner.g, int(-np.log10(env.e)-1))*100, '%')
print('Optimality gap (PI)', around(abs(OSPI_learner.g-PI_learner.g)/PI_learner.g, int(-np.log10(env.e)-1))*100, '%')

Pi = OSPI_learner.Pi
V = OSPI_learner.V
Pi = VI_learner.Pi
V = VI_learner.V

if env.J > 1:
    plot_Pi(env, PI_learner, Pi, zero_state=True)
    plot_Pi(env, PI_learner, Pi, zero_state=False)
for i in arange(env.J):
    plot_Pi(env, PI_learner, Pi, zero_state=True, i=i)
#    plot_V(env, PI_learner, V, zero_state=True, i=i)
    plot_Pi(env, PI_learner, Pi, zero_state=False, i=i)