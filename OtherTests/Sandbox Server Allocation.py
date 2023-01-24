"""
Works for any J=1 or higher.

@author: Jeroen
"""

import os
PATH = (r"D:\Programs\SURFdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")
# PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")
os.chdir(PATH)
from init import Env
import numpy as np
from numpy import array, exp, around, ones, eye, dot

from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy import optimize

np.set_printoptions(precision=3, linewidth=150, suppress=False)
# env = Env(J=3, S=10,
#           mu=np.array([0.5, 1, 0.5]),
#           lmbda=np.array([1, 1.5, 0.5]),
#           t=array([1/4]*3),
#           r=array([4]*3), c = np.array([2, 3, 1]),
#           gamma=5, D=50,
#           time_check=False)
env = Env(J=2, S=2, mu=array([0.5,1.5]), lmbda=array([1,1]), t=array([3,3]),
          r=array([1,1]), c=array([1,1]), P=0, e=1e-5, 
          gamma=2, D=10)
# env = Env(J=1, S=4, mu=array([0.5]), lmbda=array([1.5]), t=array([1]),
#           r=array([4]), c=array([5]),
#           gamma=2, D=10)

print("Total System load rho<1?", sum(env.lmbda) / (sum(env.mu) * env.S)) 
print("lambda/mu=", env.lmbda/env.mu)  # Used to estimate s_star

def get_pi_0(env, s, rho):
    """Calculate pi(0)."""
    pi_0 = s*exp(s*rho) / (s*rho)**s * gamma_fun(s)*reg_up_inc_gamma(s,s*rho)
    pi_0 += (env.gamma + rho * env.lmbda)/env.gamma * (1 / (1 - rho))
    return 1 / pi_0

def get_tail_prob(env, s, rho, pi_0):
    """P(W>t)."""
    tail_prob = pi_0/(1-rho) * \
        (env.lmbda+env.gamma) / (env.gamma + env.lmbda*pi_0) * \
        (1-(s*env.mu-env.lmbda) / (s*env.mu+env.gamma))**(env.gamma*env.t)
    return tail_prob

def server_allocation_cost(s, env):
    """Sums of g per queue, note that -reward is returned."""
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = env.a / s
        pi_0 = get_pi_0(env, s, rho)
        tail_prob = get_tail_prob(env, s, rho, pi_0)
    tail_prob[~np.isfinite(tail_prob)] = 1  # Correct dividing by 0
    res = (env.r - env.c*tail_prob)*(env.lmbda + pi_0*env.lmbda**2/env.gamma)
    return -np.sum(res, axis=len(np.shape(s))-1)

def server_allocation(env):
    """Docstring."""
    if np.all(env.t > 0):
        weighted_load = (1/env.t)/sum((1/env.t)) * \
            env.c/sum(env.c) * env.a/sum(env.a)
    else:
        weighted_load = env.c/sum(env.c) * env.a/sum(env.a)
    x0 = env.a + weighted_load / sum(weighted_load) * (env.S - sum(env.a))
    
    lb_bound = env.a  # lb <= A.dot(x) <= ub
    ub_bound = env.S - dot((ones((env.J,env.J)) - eye(env.J)), env.a)
    bounds = optimize.Bounds(lb_bound, ub_bound)
    
    A_cons = array([1]*env.J)
    lb_cons = env.S  # Equal bounds represent equality constraint
    ub_cons = env.S
    lin_cons = optimize.LinearConstraint(A_cons, lb_cons, ub_cons)
    
    s_star = optimize.minimize(server_allocation_cost, x0, args=(env), 
                               bounds=bounds, constraints=lin_cons).x
    return s_star

states = env.S_states[(np.sum(env.S_states, axis=1) == env.S) &
                      (np.min(env.S_states, axis=1) != 0)]
s_star_int = states[np.argmin(server_allocation_cost(states, env))]

s_star = server_allocation(env)

print(s_star_int, ', g:', around(-server_allocation_cost(s_star_int, env),4))
print(s_star, ', g:', around(-server_allocation_cost(s_star, env),4))
