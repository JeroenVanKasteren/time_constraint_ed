"""
Sandbox One-step policy improvement

Note that V(x) for x<=0 is used to calculate V(0), which is used in V(x) x>0.
"""

import numpy as np
from numpy import array, arange, zeros, exp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
from numba import njit

from OtherTests.init import Env
from src.Plotting import plot_Pi, plot_V

np.set_printoptions(precision=4, linewidth=150, suppress=True)

# env = Env(J=1, S=1, mu=array([2]), lmbda=array([1]), t=array([2]),
#           r=array([1]), c=array([1]), P=0,
#           gamma=2, D=50, e=1e-5, trace=False)
env = Env(J=2, S=2, lmbda=array([0.5, 0.5]), mu=array([1, 1]), t=array([1., 1.]),
          r=array([1, 1]), c=array([1, 1]), P=0,
          gamma=2, D=6, trace=True)

Not_Evaluated = env.NOT_EVALUATED
Servers_Full = env.SERVERS_FULL
None_Waiting = env.NONE_WAITING
Keep_Idle = env.KEEP_IDLE

J = env.J
S = env.S
D = env.D
gamma = env.gamma
t = env.t
c = env.c
r = env.r
P = env.P
sizes = env.sizes
size = env.size
dim = env.dim
sizes_i = env.sizes_i
size_i = env.size_i
dim_i = env.dim_i
s_states = env.s_states
x_states = env.x_states
P_xy = env.P_xy


def V_app_f(env, i):
    """Calculate V for a single queue."""
    S=env.S; lmbda=env.lmbda[i]; mu=env.mu[i]; gamma=env.gamma; rho=env.rho[i]; r=env.r[i]; a=env.a[i]; g=env.g[i]
    V = zeros(S + env.D + 1)

    x = arange(-S+1, 0+1)  # V(x) for x<=0, with V(-s)=0
    V_x_le_0 = lambda y: (1-(y/a)**(x+S))/(1-y/a)*exp(a-y)
    V[x+S] = (g-lmbda*r)/lmbda * quad_vec(V_x_le_0, a, np.inf)[0]

    frac = (S * mu + gamma) / (lmbda + gamma)
    trm = exp(a) / a ** (S - 1) * gamma_fun(S) * reg_up_inc_gamma(S, a)
    x = arange(1, env.D + 1)  # V(x) for x>0
    V[x+S] = V[S] + (S*mu*r - g) / (gamma*S*mu*(1 - rho)**2) * \
        (lmbda + gamma - lmbda*x*(rho-1) - (lmbda+gamma)*frac**x) + \
            1/(gamma*(rho-1)) * (g - S*mu*r - gamma/lmbda *
                                 (g + (g-lmbda*r)/rho * trm)) * (-rho + frac**(x-1))
    # -1_{x > gamma*t}[...]
    x = arange(gamma*env.t[i]+1, env.D+1).astype(int)
    V[x+S] -= env.c[i] / (gamma * (1 - rho)**2) * \
        (lmbda + gamma - lmbda*(x - gamma*env.t[i] - 1) * (rho - 1) -
         (lmbda + gamma) * frac**(x-gamma*env.t[i]-1))
    return V


def get_V_app(env, V_app):
    """Approximation of value function.

    Create a list V_memory with V_i(x), i=class.
    """
    for i in arange(env.J):
        V_i = V_app_f(env, i)
        for x in arange(env.D + 1):
            states = [slice(None)]*(env.J*2)
            states[i] = x
            V_app[tuple(states)] += V_i[x+S]
    return V_app


def init_W(env, V, W):
    for i in arange(env.J):
        states = np.append(i, [slice(None)] * (J * 2))
        states[1 + i] = slice(D)
        next_states = [slice(None)] * (J * 2)
        next_states[i] = slice(1, D + 1)
        W[tuple(states)] = V[tuple(next_states)]
        states[1 + i] = D
        next_states[i] = D
        W[tuple(states)] = V[tuple(next_states)] - P
    W[J] = V
    return W


# @njit
def W_f(V, W, Pi):
    """W given policy."""
    V = V.reshape(size)
    W = W.reshape(size_i)
    Pi = Pi.reshape(size_i)
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = i * sizes_i[0] + np.sum(x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                if Pi[state] > 0:
                    j = Pi[state] - 1
                    W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                    next_x = x.copy()
                    for y in arange(x[j] + 1):
                        next_x[j] = y
                        if (i < J) and (i != j):
                            next_x[i] = min(next_x[i] + 1, D)
                        next_s = s.copy()
                        next_s[j] += 1
                        next_state = np.sum(
                            next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W.reshape(dim_i)


def V_f(env, V, W):
    """V_t."""
    states_c = [slice(None)] * (env.J * 2)
    V_t = env.tau * V
    for i in arange(env.J):
        states_i = np.append(i, [slice(None)] * (env.J * 2))

        states = states_c.copy()
        next_states = states_i.copy()
        states[i] = 0  # x_i = 0
        next_states[1 + i] = 0
        V_t[tuple(states)] += env.lmbda[i] * (W[tuple(next_states)] - V[tuple(states)])

        states = states_c.copy()
        next_states = states_i.copy()
        states[i] = slice(1, env.D + 1)  # 0 < x_i <= D
        next_states[1 + i] = slice(1, env.D + 1)  # 0 < x_i <= D
        V_t[tuple(states)] += env.gamma * (W[tuple(next_states)] - V[tuple(states)])

        for s_i in arange(1, env.S + 1):  # s_i
            states = states_c.copy()
            next_states = states_i.copy()
            states[env.J + i] = s_i
            next_states[0] = env.J
            next_states[1 + env.J + i] = s_i - 1
            V_t[tuple(states)] += s_i * env.mu[i] * (W[tuple(next_states)] - V[tuple(states)])
    return V_t / env.tau


@njit
def policy_improvement(V, W, Pi):
    """Determine best action/policy per state by one-step lookahead."""
    V = V.reshape(size)
    W = W.reshape(size_i)
    Pi = Pi.reshape(size_i)
    stable = True
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = i * sizes_i[0] + np.sum(x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                pi = Pi[state]
                Pi[state] = Keep_Idle if ((np.sum(x) > 0) or (i < J)) else Pi[state]
                w = W[state]
                for j in arange(J):
                    if (x[j] > 0) or (j == i):  # Class i waiting, arrival, or time passing
                        value = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        w -= P if x[j] == D else 0
                        next_x = x.copy()
                        for y in arange(x[j] + 1):
                            next_x[j] = y
                            if (i < J) and (i != j):
                                next_x[i] = min(next_x[i] + 1, D)
                            next_s = s.copy()
                            next_s[j] += 1
                            next_state = np.sum(next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                            value += P_xy[j, x[j], y] * V[next_state]
                        Pi[state] = j + 1 if value >= w else Pi[state]
                        w = array([value, w]).max()
                if pi != Pi[state]:
                    stable = False
    return Pi.reshape(dim_i), stable


def policy_evaluation(env, V, W, Pi, name, count=0):
    """Policy Evaluation."""
    inner_count = 0
    converged = False
    while not converged:
        W = init_W(env, V, W)
        W = W_f(V, W, Pi)
        V_t = V_f(env, V, W)
        converged, g = env.convergence(V_t, V, count, name, j=inner_count)
        V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
        inner_count += 1
    return V, g


# One Step Policy Improvement
name = 'One-step Policy Improvement'
V_app = np.zeros(dim)
W = zeros(env.dim_i)
Pi = env.init_Pi()

env.timer(True, name, env.trace)
V_app = get_V_app(env, V_app)
W = init_W(env, V_app, W)
Pi, _ = policy_improvement(V_app, W, Pi)
env.timer(False, name, env.trace)

env.timer(True, name, env.trace)
V = zeros(dim)  # V_{t-1}
V, g = policy_evaluation(env, V, W, Pi, name, count=0)
env.timer(False, name, env.trace)

print("V", V)
print("Pi", Pi)
print("g", g)

# if env.J > 1:
#     plot_Pi(env, env, Pi, zero_state=True)
#     plot_Pi(env, env, Pi, zero_state=False)
# for i in arange(env.J):
#     plot_Pi(env, env, Pi, zero_state=True, i=i)
#     plot_V(env, env, V, zero_state=True, i=i)
