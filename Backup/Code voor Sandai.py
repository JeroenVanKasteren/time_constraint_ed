"""
Value Iteration.

copy() creates shallow copy (keep same references), but when applied to a
NumPy array it creates a deepcopy.
"""

import numpy as np
from numpy import arange, array, round
from itertools import product

N = 40

J = 2
s = 2
D = 2
dim = tuple(np.repeat([D+1, s+1], J))

lambda_ = array([3/4]*J)
mu = array([0.6]*J)
t = array([1.]*J)
weight = array([1.]*J)

gamma = 1
P = 1e2

alpha = t*gamma
tau = s*np.max(mu) + \
    np.sum(np.maximum(lambda_, gamma))

P_xy = np.zeros([J, D+1, D+1])
A = np.indices([D+1, D+1])  # x=A[0], y=A[1]
for i in arange(J):  # For every class
    mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
    P_xy[i, 1:, 1:][mask_tril] = (gamma / (lambda_[i] + gamma)) ** \
        (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
        lambda_[i] / (lambda_[i] + gamma)
    P_xy[i, 1:, 0] = (gamma / (lambda_[i] + gamma)) ** A[0, 1:, 0]
P_xy[:,0,0] = 1

# -------------------------------------------------------------------------- #
# Init
V = np.zeros(dim)
V_t = np.zeros(dim)
W = -np.ones(dim)

s_states = array(list(product(arange(s+1), repeat=J)))
s_states = s_states[np.sum(s_states, axis=1) <= s]
s_full = s_states[np.sum(s_states, axis=1) == s]
s_not_full = s_states[np.sum(s_states, axis=1) < s]
x_states = array(list(product(arange(D+1), repeat=J)))

for n in arange(N):
    # W
    for s_state in s_full:
        states = [slice(None)]*(J*2)
        states[slice(J, J*2)] = s_state
        W[tuple(states)] = V[tuple(states)]
        for i in arange(J):
            states = [slice(None)]*(J*2)
            states[slice(J, J*2)] = s_state
            states[i] = D
            W[tuple(states)] = P + V[tuple(states)]

    for s_state in s_not_full:
        for x_state in x_states:
            multi_state = np.ravel([x_state, s_state])
            w = V[tuple(multi_state)] + (P if any(x_state == D) else 0)
            for i in arange(J):
                if(x_state[i] > 0):  # If someone of class i waiting
                    value = 1 if x_state[i] > alpha[i] else 0
                    next_state = list(multi_state)
                    next_state[i] = slice(x_state[i] + 1)  # x_i
                    next_state[J+i] += 1  # s_i
                    value += np.sum(P_xy[i, x_state[i], slice(x_state[i] + 1)]
                                    * V[tuple(next_state)])
                    w = np.min([value, w])
            W[tuple(multi_state)] = w

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V.copy()
    for i in arange(J):
        # x_i = 0
        states = all_states.copy()
        next_states = all_states.copy()
        states[i] = 0
        next_states[i] = 1
        V_t[tuple(states)] += lambda_[i] * (W[tuple(next_states)] -
                                            V[tuple(states)])
        # 0 < x_i < D
        states = all_states.copy()
        next_states = all_states.copy()
        states[i] = slice(1, D)
        next_states[i] = slice(2, D+1)
        V_t[tuple(states)] += gamma * (W[tuple(next_states)] -
                                       V[tuple(states)])
        # x_i = D
        states = all_states.copy()
        states[i] = D
        V_t[tuple(states)] += gamma * (W[tuple(states)] -
                                       V[tuple(states)])
        # s_i
        for s_i in arange(1, s+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    # Convergence check
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max
    for s_state in s_states:
        states = [slice(None)]*(J*2)
        states[slice(J, J*2)] = s_state
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.max(diff), delta_min])
    # print("g: ", round((delta_max + delta_min)/2 * tau, 2),
    print("delta: ", round((delta_max - delta_min)),
          'D_min', round(delta_min,2), 'D_max', round(delta_max,2))

    # For next iteration
    V -= V[tuple([0]*(J*2))]
    V = V_t.copy()/tau
