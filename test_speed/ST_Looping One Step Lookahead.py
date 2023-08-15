"""
Take home.
Small instances do deviate, numba slower, but those times are neglegtable

J = 3; S = 4; D = 30
Test 1, original code (Python only)
 time w 249.4108
 time V 1.7901
0.0
Test 1 takes too long.

J = 3; S = 5; D = 40
Test 2, W and V numba functions
 time w 41.1354
 time V 7.2274
0.0
Test 3,  w Numba, V python
 time w 40.4727
 time V 1.0503
0.0
Test 4, w Numba & Python, V python
 time w 39.9215
 time V 1.0823
0.0
Test 5, w Numba & Python, reshape outside Numba, V python
 time w 40.3392
 time V 1.1078
0.0
Test Final code, V Python & W numba (meest overzichtelijk)
 time w 40.8787
 time V 1.0888
0.0

Gekozen voor final code doordat deze het minste is in # regels en er
nauwelijks verschil zit tussen test 3, 4, 5, en 6
let erop dat test 2 beduidend langzamer is in het berekenen van V!

Numba fixes global variables when function is compiled/runned

copy() creates shallow copy (keep same references), but when applied to a
NumPy array it creates a deepcopy.
A deepcopy is also created by calculations of numpy arrays X = b*A
"""

import numpy as np
from numpy import arange, array, round
from itertools import product
from timeit import default_timer
from numba import njit
np.set_printoptions(suppress=True)

N = 2

J = 3
S = 5
D = 5
dim = tuple(np.repeat([D+1, S+1], J))
size = np.prod(dim)

lambda_ = array([3/4]*J)
mu = array([0.6]*J)
t = array([1.]*J)
weight = array([1.]*J)

gamma = 1
P = 1e2
e = 0.01

alpha = t*gamma
tau = S*np.max(mu) + \
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

sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

s_states = array(list(product(arange(S+1), repeat=J)))
s_states = s_states[np.sum(s_states, axis=1) <= S]
s_full = s_states[np.sum(s_states, axis=1) == S]
s_not_full = s_states[np.sum(s_states, axis=1) < S]

x_states = array(list(product(arange(D+1), repeat=J)))
x_penalty = [slice(None)]*(J*2)
x_penalty[slice(J)] = x_states[np.any(x_states==D, axis=1)].T
x_no_penalty = [slice(None)]*(J*2)
x_no_penalty[slice(J)] = x_states[np.invert(np.any(x_states==D, axis=1))].T

# -------------------------------------------------------------------------- #
# Test 1, original code
V = np.zeros(dim)
W = np.zeros(dim)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing
    for s in s_full:
        states = [slice(None)]*(J*2)
        states[slice(J, J*2)] = s
        W[tuple(states)] = V[tuple(states)]
        for i in arange(J):
            states = [slice(None)]*(J*2)
            states[slice(J, J*2)] = s
            states[i] = D
            W[tuple(states)] = P + V[tuple(states)]

    for s in s_not_full:
        for x in x_states:
            multi_state = np.ravel([x, s])
            w = V[tuple(multi_state)] + (P if any(x == D) else 0)
            for i in arange(J):
                if(x[i] > 0):  # If someone of class i waiting
                    value = 1 if x[i] > alpha[i] else 0
                    next_state = list(multi_state)
                    next_state[i] = slice(x[i] + 1)  # x_i
                    next_state[J+i] += 1  # s_i
                    value += np.sum(P_xy[i, x[i], slice(x[i] + 1)]
                                    * V[tuple(next_state)])
                    w = np.min([value, w])
            W[tuple(multi_state)] = w
    timing_w += default_timer() - start_w  # Timing

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V
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
        for s_i in arange(1, S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    V_t = V_t/tau
    # Convergence check of only valid states
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = all_states.copy()
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        if delta_max - delta_min > e:
            break
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[tuple([0]*(J*2))]
    # print('n', n)
    # print('W', round(W,2))
    # print('V', round(V,2))

print('Test 1, original code \n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
print(V[tuple(np.ravel([x_states[1], s_states[0]]))])

# -------------------------------------------------------------------------- #
# Test 2, W and V numba functions
@njit
def numba_w(V, W):
    """Docstring."""
    for s in s_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            W[state] = V[state] + (P if np.any(x == D) else 0)

    for s in s_not_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            w = V[state] + (P if np.any(x == D) else 0)
            for i in arange(J):
                if(x[i] > 0):  # If someone of class i waiting
                    value = 1 if x[i] > alpha[i] else 0
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i], y] * V[next_state]
                    w = array([value, w]).min()
            W[state] = w

@njit
def numba_V(V, W):
    """Docstring."""
    V_t = tau * V.copy()
    for x in x_states:
        for s in s_states:
            state = np.sum(x * sizes[0:J] + s * sizes[J:J*2])
            for i in arange(J):
                f = lambda_[i] if x[i] == 0 else gamma
                next_x = x.copy()
                next_x[i] = array([x[i]+1, D]).min()
                next_state = np.sum(next_x * sizes[0:J] + s * sizes[J:J*2])
                V_t[state] += f * (W[next_state] - V[state])
                next_s = s.copy()
                next_s[i] = array([s[i]-1, 0]).max()
                next_state = np.sum(x * sizes[0:J] + next_s * sizes[J:J*2])
                V_t[state] += s[i] * mu[i] * (W[next_state] - V[state])
    return V_t/tau

V = np.zeros(size)
W = np.zeros(size)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing
    numba_w(V, W)
    timing_w += default_timer() - start_w  # Timing

    V_t = numba_V(V, W)

    # Convergence check
    diff = np.abs(V_t - V)
    delta_max = np.max(diff)
    delta_min = np.min(diff)
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[0]
    # print('n', n)
    # print('W', round(W.reshape(dim),2))
    # print('V', round(V.reshape(dim),2))

print('Test 2, W and V numba functions \n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
state = np.sum(x_states[1] * sizes[0:J] + s_states[0] * sizes[J:J*2])
print(np.mean(V[state]))

# -------------------------------------------------------------------------- #
# Test 3, w Numba, V python
@njit
def numba_w(V, W):
    """Docstring."""
    for s in s_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            W[state] = V[state] + (P if np.any(x == D) else 0)

    for s in s_not_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            w = V[state] + (P if np.any(x == D) else 0)
            for i in arange(J):
                if(x[i] > 0):  # If someone of class i waiting
                    value = 1 if x[i] > alpha[i] else 0
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i], y] * V[next_state]
                    w = array([value, w]).min()
            W[state] = w
    return V, W

V = np.zeros(dim)
W = np.zeros(dim)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing
    V = V.reshape(size)
    W = W.reshape(size)
    V, W = numba_w(V, W)
    V = V.reshape(dim)
    W = W.reshape(dim)
    timing_w += default_timer() - start_w  # Timing

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V
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
        for s_i in arange(1, S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    V_t = V_t/tau
    # Convergence check of only valid states
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = all_states.copy()
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        if delta_max - delta_min > e:
            break
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[tuple([0]*(J*2))]
    # print('n', n)
    # print('W', round(W,2))
    # print('V', round(V,2))

print('Test 3,  w Numba, V python \n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
print(V[tuple(np.ravel([x_states[1], s_states[0]]))])

# -------------------------------------------------------------------------- #
# Test 4, w Numba & Python, V python
@njit
def numba_w(V, W):
    """Docstring."""
    V = V.reshape(size)
    W = W.reshape(size)
    for s in s_not_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            w = V[state] + (P if np.any(x == D) else 0)
            for i in arange(J):
                if(x[i] > 0):  # If someone of class i waiting
                    value = 1 if x[i] > alpha[i] else 0
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i], y] * V[next_state]
                    w = array([value, w]).min()
            W[state] = w
    return V.reshape(dim), W.reshape(dim)

V = np.zeros(dim)
W = np.zeros(dim)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing

    for s in s_states:
        states = x_penalty.copy()
        states[slice(J,J*2)] = s
        W[tuple(states)] = V[tuple(states)] + P

        states = x_no_penalty.copy()
        states[slice(J,J*2)] = s
        W[tuple(states)] = V[tuple(states)]

    V, W = numba_w(V, W)
    timing_w += default_timer() - start_w  # Timing

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V
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
        for s_i in arange(1, S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    V_t = V_t/tau
    # Convergence check of only valid states
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = all_states.copy()
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        if delta_max - delta_min > e:
            break
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[tuple([0]*(J*2))]
    # print('n', n)
    # print('W', round(W,2))
    # print('V', round(V,2))

print('Test 4, w Numba & Python, V python\n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
print(V[tuple(np.ravel([x_states[1], s_states[0]]))])

# -------------------------------------------------------------------------- #
# Test 5, w Numba & Python, reshape outside Numba, V python
@njit
def numba_w(V, W):
    """Docstring."""
    for s in s_not_full:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            w = W[state]
            for i in arange(J):
                if(x[i] > 0):  # If someone of class i waiting
                    value = 1 if x[i] > alpha[i] else 0
                    for y in arange(x[i] + 1):
                        next_x = x.copy()
                        next_x[i] = y
                        next_s = s.copy()
                        next_s[i] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        value += P_xy[i, x[i], y] * V[next_state]
                    w = array([value, w]).min()
            W[state] = w
    return V, W

V = np.zeros(dim)
W = np.zeros(dim)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing

    for s in s_states:
        states = x_penalty.copy()
        states[slice(J,J*2)] = s
        W[tuple(states)] = V[tuple(states)] + P

        states = x_no_penalty.copy()
        states[slice(J,J*2)] = s
        W[tuple(states)] = V[tuple(states)]

    V = V.reshape(size)
    W = W.reshape(size)
    V, W = numba_w(V, W)
    V = V.reshape(dim)
    W = W.reshape(dim)
    timing_w += default_timer() - start_w  # Timing

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V
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
        for s_i in arange(1, S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    V_t = V_t/tau
    # Convergence check of only valid states
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = all_states.copy()
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        if delta_max - delta_min > e:
            break
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[tuple([0]*(J*2))]
    # print('n', n)
    # print('W', round(W,2))
    # print('V', round(V,2))

print('Test 5, w Numba & Python, reshape outside Numba, V python\n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
print(V[tuple(np.ravel([x_states[1], s_states[0]]))])

# -------------------------------------------------------------------------- #
# Test Final code, W numba & Python, V Python (meest overzichtelijk)
@njit
def numba_w(V, W):
    """Docstring."""
    V = V.reshape(size)
    W = W.reshape(size)
    for s in s_states:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            W[state] = V[state] + (P if np.any(x == D) else 0)
            if np.sum(s) < S:
                for i in arange(J):
                    if(x[i] > 0):  # If someone of class i waiting
                        value = 1 if x[i] > alpha[i] else 0  # Cost
                        for y in arange(x[i] + 1):
                            next_x = x.copy()
                            next_x[i] = y
                            next_s = s.copy()
                            next_s[i] += 1
                            next_state = np.sum(next_x*sizes[0:J] + \
                                                next_s*sizes[J:J*2])
                            value += P_xy[i, x[i], y] * V[next_state]
                        W[state] = array([value, W[state]]).min()
    return V.reshape(dim), W.reshape(dim)

V = np.zeros(dim)
W = np.zeros(dim)

start = default_timer()
timing_w = 0
for n in arange(N):
    # W
    start_w = default_timer()  # Timing
    V, W = numba_w(V, W)
    timing_w += default_timer() - start_w  # Timing

    # V_t
    all_states = [slice(None)]*(J*2)
    V_t = tau * V
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
        for s_i in arange(1, S+1):
            states = all_states.copy()
            next_states = all_states.copy()
            states[J+i] = s_i
            next_states[J+i] = s_i - 1
            V_t[tuple(states)] += s_i * mu[i] * (W[tuple(next_states)] -
                                                 V[tuple(states)])
    V_t = V_t/tau
    # Convergence check of valid states only
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = all_states.copy()
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        if delta_max - delta_min > e:
            break
    if delta_max - delta_min < e:
        print("iter: ", n,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round((delta_max + delta_min)/(2 * tau), 2))
        break
    V = V_t - V_t[tuple([0]*(J*2))]
    # print('n', n)
    # print('W', round(W,2))
    # print('V', round(V,2))

print('Test Final code, V Python & W numba (meest overzichtelijk)\n',
      'time w', round(timing_w/N, 4), '\n',
      'time V', round((default_timer()-start-timing_w)/N, 4))
print(V[tuple(np.ravel([x_states[1], s_states[0]]))])