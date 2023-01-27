"""
Sandbox Policy Iteration
"""

import numpy as np
from numba import njit
from numpy import array, arange, zeros

from OtherTests.init import Env
from src.Plotting import plot_Pi, plot_V

np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(J=1, S=1, mu=array([2]), lmbda=array([1]), t=array([2]),
          r=array([1]), c=array([1]), P=0,
          gamma=10, D=100, e=1e-5, trace=False)
# env = Env(J=2, S=2, lmbda=array([0.5,0.5]), mu=array([1,1]), t=array([1.]),
#           r=array([1,1]), c=array([1,1]), P=0,
#           gamma=5, D=5, trace=True)

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


@njit
def W_f(V, W, Pi):
    """W given policy."""
    V = V.reshape(size)
    W = W.reshape(size_i)
    Pi = Pi.reshape(size_i)
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = np.sum(i * sizes_i[0] + x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                if Pi[state] != Keep_Idle:
                    j = Pi[state] - 1
                    W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                    next_x = x.copy()
                    for y in arange(x[j] + 1):
                        next_x[j] = y
                        if (i < J) and (i != j):
                            next_x[i] += 1
                        next_s = s.copy()
                        next_s[j] += 1
                        next_state = np.sum(
                            next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W.reshape(dim_i)


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


# Policy Iteration
name = 'Policy Iteration'
W = zeros(env.dim)
Pi = env.init_Pi()

count = 0
unstable = True

env.timer(True, name, env.trace)
while unstable:
    V = zeros(env.dim)  # V_{t-1}
    V, g = policy_evaluation(env, V, W, Pi, 'Policy Evaluation of PI', count)
    # TODO switch unstable to stable!
    Pi, unstable = policy_improvement(V, Pi)
    if count > env.max_iter:
        break
    count += 1
env.timer(False, name, env.trace)

# print("V", V)
# print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_Pi(env, env, Pi, zero_state=True)
    plot_Pi(env, env, Pi, zero_state=False)
for i in arange(env.J):
    plot_Pi(env, env, Pi, zero_state=True, i=i)
    plot_Pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_V(env, env, V, zero_state=True, i=i)

