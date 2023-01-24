"""
Sandbox One-step policy improvement

Note that V(x) for x<=0 is used to calculate V(0), which is used in V(x) x>0.


# Env initialization
J = 2
S = 4

lambda_ = np.array([1, 1])
mu = np.array([0.5, 1.5])
print(sum(lambda_/mu)/S)  # Total system load < 1
print(lambda_/mu)  # Used to estimate s_star

t = np.array([1/5]*J)
c = np.array([2, 3, 1])

gamma = 30
D = 15
P = 1e2
e = 1e-5

@author: Jeroen

Policy , Essential to have the numbers in ascending order
Pi = -5, not evalated
Pi = -1, Servers full
Pi = 0, No one waiting
Pi = i, take queue i into serves, i={1,...,J}
Pi = J+1, Keep idle

Penalty is an incentive to take people into service, it is not needed when
there is no decision (servers full)
https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
"""

PATH = (r"D:\Programs\Surfdrive\Surfdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")
# PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")

import os
os.chdir(PATH+"\Other")
from init import Env
os.chdir(PATH)
from Plotting import plot_Pi, plot_V

import numpy as np
from numpy import array, arange, exp, zeros
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
from numba import njit

np.set_printoptions(precision=3, linewidth=150, suppress=True)
# env = Env(J=1, S=2, mu=array([1]), lmbda=array([1.5]), t=array([3]),
#           r=array([1]), c=array([1]), P=0, e=1e-5,
#           gamma=2, D=15)
env = Env(J=2, S=2, mu=array([1,2]), lmbda=array([1,1]), t=array([3,3]),
          r=array([1,1]), c=array([1,1]), P=0, e=1e-5, 
          gamma=2, D=10)
# env = Env(J=2, S=3, mu=array([1,1]), lmbda=array([1,1]), t=array([3,3]),
#           r=array([1,1]), c=array([1,1]), P=0, e=1e-4,
#           gamma=2, D=20, max_iter=1e5, trace=False)

Not_Evaluated = env.NOT_EVALUATED
Servers_Full = env.SERVERS_FULL
None_Waiting = env.NONE_WAITING
Keep_Idle = env.KEEP_IDLE

def initialize_pi(env):
    "Take the longest waiting queue into service. At tie take the last queue."
    Pi = Not_Evaluated*np.ones(env.dim, dtype=int)   
    for s in env.S_states:
        states = [slice(None)]*(env.J*2)
        states[slice(env.J,env.J*2)] = s
        
        states_0 = states.copy()
        states_0[slice(0,env.J)] = [0]*env.J
        Pi[tuple(states_0)] = None_Waiting        
        if np.sum(s) == env.S:
            Pi[tuple(states)] = Servers_Full            
        else:
            for i in arange(env.J):
                states_ = states.copy()
                for x in arange(1, env.D+1):
                    states_[i] = x
                    for j in arange(env.J):
                        if j != i:
                            states_[j] = slice(0, x+1)
                    Pi[tuple(states_)] = i + 1
    return Pi

J=env.J; S=env.S; D=env.D; gamma=env.gamma; t=env.t; c=env.c; r=env.r; P=env.P
sizes=env.sizes; size=env.size; S_states=env.S_states; x_states=env.x_states
dim=env.dim; P_xy=env.P_xy

def V_app_f(env, i):
    """Calculate V for a single queue."""
    s=env.s_star[i]; lmbda=env.lmbda[i]; mu=env.mu[i]; rho=env.rho[i]; 
    a=env.a[i]; r=env.r[i]; c=env.c[i]; t=env.t[i]; g=env.g[i]
    V_i = zeros(env.D + 1)

    # V(x) for x<=0, with V(-s)=0
    V_x_le_0 = lambda y: (1-(y/a)**(s))/(1-y/a)*exp(a-y)
    V_i[0] = (g - lmbda*r)/lmbda * quad_vec(V_x_le_0, a, np.inf)[0]

    # V(x) for x>0
    x = arange(1, env.D+1) 
    frac = (s*mu + env.gamma) / (lmbda + env.gamma)
    trm = exp(a)/a**(s-1) * gamma_fun(s)*reg_up_inc_gamma(s,a)
    V_i[x] = V_i[0] + (s*mu*r - g) / (env.gamma*s*mu*(1 - rho)**2) * \
        (lmbda + env.gamma - lmbda*x*(rho-1) - (lmbda+env.gamma)*frac**x) + \
            1/(env.gamma*(rho-1)) * (g - s*mu*r - env.gamma/lmbda *(
                g + (g-lmbda*r)/rho * trm)) * (-rho + frac**(x-1))
    # -1_{x > gamma*t}[...]
    x = arange(env.gamma*t + 1, env.D + 1).astype(int)
    V_i[x] -= c / (env.gamma * (1 - rho)**2) * \
        (lmbda + env.gamma - lmbda*(x - env.gamma*t - 1) * \
         (rho - 1) - (lmbda + env.gamma) * frac**(x-env.gamma*t-1))
    return V_i

def get_V_app(env, V_app):
    """Approximation of value function.

    Create a list V_memory with V_ij(x), i=class, j=#servers for all x.
    Note only j = s*_i, ..., s will be filled, rest zero
    """
    for i in arange(env.J):
        V_i = V_app_f(env, i)
        for x in arange(env.D + 1):
            states = [slice(None)]*(env.J*2)
            states[i] = x
            V_app[tuple(states)] += V_i[x]
    return V_app

@njit
def W_f(V, W, Pi):
    """W."""
    V = V.reshape(size); W = W.reshape(size); Pi = Pi.reshape(size)
    for s in S_states:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            i = Pi[state]
            W[state] = V[state]
            if (np.sum(s) < S) & (x[i-1] > 0):  # serv. idle, any waiting
                if i == Keep_Idle:
                    W[state] = W[state] - P if np.any(x == D) else W[state]
                else:
                    i = i-1
                    W[state] = r[i] - c[i] if x[i] > gamma*t[i] else r[i]
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        W[state] += P_xy[i, x[i], y] * V[next_state]
    return W.reshape(dim)

def V_f(env, V, W):
    """V_t."""
    all_states = [slice(None)]*(env.J*2)
    V_t = env.tau * V
    for i in arange(env.J):
        # x_i = 0
        states = all_states.copy()
        next_states = all_states.copy()
        states[i] = 0  # x_i (=0)
        next_states[i] = 1  # x_i + 1
        V_t[tuple(states)] += env.lmbda[i] * (W[tuple(next_states)] -
                                              V[tuple(states)])
        # 0 < x_i < D
        states = all_states.copy()
        next_states = all_states.copy()
        states[i] = slice(1, env.D)  # x_i
        next_states[i] = slice(2, env.D+1)  # x_i + 1
        V_t[tuple(states)] += env.gamma * (W[tuple(next_states)] -
                                           V[tuple(states)])
        # x_i = D
        states = all_states.copy()
        states[i] = env.D  # x_i = D
        V_t[tuple(states)] += env.gamma * (W[tuple(states)] -
                                           V[tuple(states)])
        # s_i
        for s_i in arange(1, env.S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[env.J+i] = s_i
            next_states[env.J+i] = s_i - 1
            V_t[tuple(states)] += s_i * env.mu[i] * \
                (W[tuple(next_states)] - V[tuple(states)])
    return V_t/env.tau

@njit
def policy_improvement(V, Pi):
    """Determine best action/policy per state by one-step lookahead."""
    V = V.reshape(size)
    Pi = Pi.reshape(size)
    unstable = False
    for s in S_states:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            if np.sum(x) == 0:
                Pi[state] = None_Waiting
                continue
            if np.sum(s) == S:
                Pi[state] = Servers_Full
                continue
            pi = Pi[state]
            w = V[state] - P if np.any(x == D) else V[state]
            Pi[state] = Keep_Idle
            for i in arange(J):
                if(x[i] > 0):  # FIL class i waiting
                    value = r[i] - c[i] if x[i] > gamma*t[i] else r[i]
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i], y] * V[next_state]
                    Pi[state] = i + 1 if round(value,10) > round(w,10) else Pi[state]
                    w = array([value, w]).max()
            if pi != Pi[state]:
                unstable = True
    return Pi.reshape(dim), unstable

def policy_evaluation(env, V, W, Pi, name, count=0):
    """Policy Evaluation."""
    inner_count = 0
    while True:
        W = W_f(V, W, Pi)
        V_t = V_f(env, V, W)
        converged, g = env.convergence(V_t, V, count, name, j=inner_count)
        if(converged):
            break  # Stopping condition

        # Rescale and Save V_t
        V = V_t - V_t[tuple([0]*(J*2))]
        inner_count += 1
    return V, g

# One Step Policy Improvement
name = 'One-step Policy Improvement'
V_app = np.zeros(dim)
V = zeros(dim)  # V_{t-1}
W = zeros(dim)
Pi = initialize_pi(env)

env.timer(True, name, env.trace)
V_app = get_V_app(env, V_app)
Pi, _ = policy_improvement(V_app, Pi)
env.timer(False, name, env.trace)

env.timer(True, name, env.trace)
V, g = policy_evaluation(env, V, W, Pi, name, count=0)
env.timer(False, name, env.trace)

# print("V", V)
# print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_Pi(env, env, Pi, zero_state=True)
    plot_Pi(env, env, Pi, zero_state=False)
for i in arange(env.J):
    plot_Pi(env, env, Pi, zero_state=True, i=i)
    plot_V(env, env, V, zero_state=True, i=i)
