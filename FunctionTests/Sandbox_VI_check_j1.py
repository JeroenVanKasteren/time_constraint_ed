"""
Sandbox Value Iteration

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
"""

import numpy as np
from numba import njit
from numpy import array, arange, zeros

from OtherTests.init import Env
from src.Plotting import plot_Pi, plot_V

np.set_printoptions(precision=3, threshold=2000, linewidth=150)

# np.set_printoptions(precision=3, linewidth=150, suppress=True)
env = Env(J=1, S=1, mu=array([5]), lmbda=array([3]), t=array([2]),
          r=array([1]), c=array([1]), P=0,
          gamma=1, D=20, trace=False)
# env = Env(J=1, S=4, mu=array([1/2]), lmbda=array([1]), t=array([1]),
#           r=array([2]), c=array([1]), P=0,
#           gamma=5, D=20, trace=False, print_modulo=20)
# env = Env(J=2, S=2, lmbda=array([0.5,0.5]), mu=array([1,1]), t=array([1.]),
#           r=array([1,1]), c=array([1,1]), P=0,
#           gamma=5, D=5, trace=True)
# env = Env(J=2, S=3, mu=array([1,1]), lmbda=array([1,1]), t=array([3,3]),
#           r=array([1,1]), c=array([1,1]), P=0, e=1e-4,
#           gamma=2, D=20, max_iter=1e5, trace=False)  # g.2.22
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
        W[tuple(states)] = V[tuple(next_states)]
    W[J] = V
    return W


@njit
def W_f(V, W):
    """W."""
    V = V.reshape(size)
    W = W.reshape(size_i)
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = np.sum(i * sizes_i[0] + x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                for j in arange(J):
                    if (x[j] > 0) or (j == i):  # Class i waiting, arrival, or time passing
                        w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        # w -= P if x[j] == D else 0
                        next_x = x.copy()
                        for y in arange(x[j] + 1):
                            next_x[j] = y
                            if (i < J) and (i != j):
                                next_x[i] += 1
                            next_s = s.copy()
                            next_s[j] += 1
                            next_state = np.sum(next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                            w += P_xy[j, x[j], y] * V[next_state]
                        # W[state] = array([w, W[state]]).max() # TODO
                        W[state] = w
    return W.reshape(dim_i)
# V = V.reshape(dim)
# W = W.reshape(dim_i)


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
    V = V.reshape(size);
    W = W.reshape(size_i);
    Pi = Pi.reshape(size_i)
    stable = True
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = np.sum(i * sizes_i[0] + x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                pi = Pi[state]
                Pi[state] = Keep_Idle if (np.sum(x) or (i < J)) else Pi[state]
                w = W[state]
                for j in arange(J):
                    if (x[j] > 0) or (j == i):  # Class i waiting, arrival, or time passing
                        value = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        # w -= P if x[j] == D else 0
                        next_x = x.copy()
                        for y in arange(x[j] + 1):
                            next_x[j] = y
                            if (i < J) and (i != j):
                                next_x[i] += 1
                            next_s = s.copy()
                            next_s[j] += 1
                            next_state = np.sum(next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                            value += P_xy[j, x[j], y] * V[next_state]
                        Pi[state] = j + 1 if value > w else Pi[state]
                        # w = array([value, w]).max()  # TODO
                        # Pi[state] = j + 1
                        w = value
                if pi != Pi[state]:
                    stable = False
    return Pi.reshape(dim_i), stable

# Value Iteration
name = 'Value Iteration'
V = zeros(env.dim)  # V_{t-1}
W = zeros(env.dim_i)
Pi = env.init_Pi()

V_t_ = zeros(env.dim)  # TODO
W_ = zeros(env.dim_i)

count = 0
env.timer(True, name, env.trace)
converged = False
while not converged:  # Update each state.
    W = init_W(env, V, W)

    for i in arange(J):
        W_[0, 0:D, :] = V[1:D + 1, :].copy()  # TODO
        W_[0, D, :] = V[D, :]
        W_[1, :, :] = V.copy()
    if not np.allclose(W_, W):
        print("W_init incorrect")
        print("Iter: ", count)
        print("W_", W_)
        print("W", W)
        break

    W = W_f(V, W)

    W_[0, 0, 0] = max(r[0] + V[0, 1], V[1, 0])  # TODO
    W_[1, 0, 0] = 0  # V[0, 0]  # TODO
    for x in arange(1, D + 1):
        W_[0, x, 0] = r[0] - c[0] if x > gamma * t[0] else r[0]
        W_[1, x, 0] = r[0] - c[0] if x > gamma * t[0] else r[0]
        for y in arange(x + 1):
            W_[0, x, 0] += P_xy[0, x, y] * V[y, 1]
            W_[1, x, 0] += P_xy[0, x, y] * V[y, 1]
        W_[0, x, 0] = max(W_[0, x, 0], V[min(D, x + 1), 0])
        W_[1, x, 0] = max(W_[1, x, 0], V[x, 0])
    W_[0, 0:D, 1] = V[1:D+1, 1]
    W_[0, D, 1] = V[D, 1]
    W_[1, :, 1] = V[:, 1]
    if not np.allclose(W_, W):
        print("W incorrect")
        print("Iter: ", count)
        print("W_", W_)
        print("W", W)
        break

    V_t = V_f(env, V, W)

    V_t_[0, 0] = env.lmbda[0] * W[0, 0, 0] + (env.tau - env.lmbda[0]) * V[0, 0]  # TODO
    V_t_[0, 1] = env.lmbda[0] * W[0, 0, 1] + env.mu[0] * W[1, 0, 0] + (env.tau - env.lmbda[0] - env.mu[0]) * V[0, 1]
    V_t_[1:D+1, 0] = gamma * W[0, 1:D + 1, 0] + (env.tau - gamma) * V[1:D+1, 0]
    V_t_[1:D+1, 1] = gamma * W[0, 1:D + 1, 1] + env.mu[0] * W[1, 1:D + 1, 0] + (env.tau - gamma - env.mu[0]) * V[1:D + 1, 1]
    if not np.allclose(V_t_, V_t * env.tau):
        # V_t_== V_t*env.tau
        print("V_t incorrect")
        print("Iter: ", count)
        print("V_t_", np.round(V_t_, 5))
        print("V_t", V_t * env.tau)
        break

    converged, g = env.convergence(V_t, V, count, name)
    V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
    count += 1

env.timer(False, name, env.trace)

# Determine policy via Policy Improvement.
W = init_W(env, V, W)
Pi, _ = policy_improvement(V, W, Pi)

print("V", V)
print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_Pi(env, env, Pi, zero_state=True)
    plot_Pi(env, env, Pi, zero_state=False)
for i in arange(env.J):
    plot_Pi(env, env, Pi, zero_state=True, i=i)
    plot_Pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_V(env, env, V, zero_state=True, i=i)

