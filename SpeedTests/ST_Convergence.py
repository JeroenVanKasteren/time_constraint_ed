"""
Convergence check of only valid states.

Take home.
Note that Test 1 and 5 can even be optimized by checking every loop if
the convergence criteria failed. This improves the speed a lot!
        # if delta_max - delta_min > e:
        #     break

Test 1, python code
 time 0.4453
91733729.0 0.0
Test 2, python code vectorized (by setting all invalid states to 0)
 time 1.4224
91733729.0 0.0
Test 3, one-dim python (=test 2, with no invalid state operation)
 time V 0.6589
91733729.0 0.0
Test 4, Numba
 time V 8.3577
91733729.0 0.0
Test 5, one-dim python in combi with test 1
 time 0.4718
91733729.0 0.0
"""

import numpy as np
from numpy import arange, array, round
from itertools import product
from timeit import default_timer
from numba import njit
np.set_printoptions(suppress=True)

N = 3

J = 3
S = 10
D = 40
dim = tuple(np.repeat([D+1, S+1], J))
size = np.prod(dim)



sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

s_states = array(list(product(arange(S+1), repeat=J)))
invalid_s = s_states[np.sum(s_states, axis=1) > S]
s_states = s_states[np.sum(s_states, axis=1) <= S]
x_states = array(list(product(arange(D+1), repeat=J)))

# -------------------------------------------------------------------------- #
# Test 1, python code partially vectorized
V = np.arange(size).reshape(dim)
V_t = np.ones(dim)
e = np.mean(V)

start = default_timer()
for n in arange(N):
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = [slice(None)]*(J*2)
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        # if delta_max - delta_min > e:
        #     break
print('Test 1, python code \n',
      'time', round((default_timer()-start)/N, 4))
print(round(delta_max, 2), round(delta_min, 2))

# -------------------------------------------------------------------------- #
# Test 2, python code vectorized (by setting all invalid states to 0)
V = np.arange(size).reshape(dim)
V_t = np.ones(dim)


start = default_timer()
for n in arange(N):
    states = [slice(None)]*(J*2)
    states[slice(J,J*2)] = invalid_s.T
    V[tuple(states)] = 0
    V_t[tuple(states)] = 0

    diff = np.abs(V_t - V)
    delta_max = np.max(diff)
    delta_min = np.min(diff)

print('Test 2, python code vectorized (by setting all invalid states to 0)\n',
      'time', round((default_timer()-start)/N, 4))
print(round(delta_max, 2), round(delta_min, 2))

# -------------------------------------------------------------------------- #
# Test 3, one-dim python
V = np.arange(size).reshape(dim)
V_t = np.ones(dim)
states = [slice(None)]*(J*2)
states[slice(J,J*2)] = invalid_s.T
V[tuple(states)] = V_t[tuple(states)]
V = V.reshape(size)
V_t = V_t.reshape(size)

start = default_timer()  # Timing
for n in arange(N):
    # Convergence check
    diff = np.abs(V_t - V)
    delta_max = np.max(diff)
    delta_min = np.min(diff)

print('Test 3, one-dim python (=test 2, with no invalid state operation)\n',
      'time V', round((default_timer()-start)/N, 4))
print(round(delta_max, 2), round(delta_min, 2))

# -------------------------------------------------------------------------- #
# Test 4, Numba
V = np.arange(size)
V_t = np.ones(size)

@njit
def numba_conv(V, V_t):
    """Docstring."""
    diff = np.abs(V_t[0] - V[0])
    max_min = array([diff, diff])
    for s_state in s_states:
        for x_state in x_states:
            state = np.sum(x_state*sizes[0:J] + s_state*sizes[J:J*2])
            diff = np.abs(V_t[state] - V[state])
            max_min[0] = array([diff, max_min[0]]).max()
            max_min[1] = array([diff, max_min[1]]).min()
    return max_min

start = default_timer()  # Timing
for n in arange(N):
    # Convergence check
    delta_max, delta_min = numba_conv(V, V_t)
print('Test 4, Numba\n',
      'time V', round((default_timer()-start)/N, 4))
print(round(delta_max, 2), round(delta_min, 2))

# -------------------------------------------------------------------------- #
# Test 5, one-dim python in combi with test 1
V = np.arange(size).reshape(dim)
V_t = np.ones(dim)
states = [slice(None)]*(J*2)
states[slice(J,J*2)] = invalid_s.T
V[tuple(states)] = V_t[tuple(states)]
V = V.reshape(size)
V_t = V_t.reshape(size)

start = default_timer()
for n in arange(N):
    V = V.reshape(dim)
    V_t = V_t.reshape(dim)
    delta_max = V_t[tuple([0]*(J*2))] - V[tuple([0]*(J*2))]
    delta_min = delta_max.copy()
    for s in s_states:
        states = [slice(None)]*(J*2)
        states[slice(J,J*2)] = s
        diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
        delta_max = np.max([np.max(diff), delta_max])
        delta_min = np.min([np.min(diff), delta_min])
        # if delta_max - delta_min > e:
        #     break
    V = V.reshape(size)
    V_t = V_t.reshape(size)
print('Test 5, one-dim python in combi with test 1 \n',
      'time', round((default_timer()-start)/N, 4))
print(round(delta_max, 2), round(delta_min, 2))
