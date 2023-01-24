"""
SpeedTest (ST).
Similarities with ST_Looping

For very small dim (J=2, s=2, D=3), numba slower
For medium dim (J=2, s=5, D=10), numba faster

Note, Itertools loops over all states, while all other's skip invalid states
Note, append is very slow, define size up front!
Again, if code can be (partially) smartly optimized, this is always faster.

For large dim (J=3, s=5, D=25), numba faster
Test 1, Itertools (product)
 creation 0.0 time 20.0433
360.0
Test 2, for loop premade x, s vector
 creation 0.0145 time 4.495
360.0
Test 3, for loop premade x, s vector, prefilter invalid states
 creation 0.0152 time 4.5661
360.0
Test 4, Numba premade x, s vector
 creation 0.0298 time 0.1619
360.0
Test 5, Numba premade x,s, prefilter invalid states
 creation 0.016 time 0.1747
360.0
Test 6, Numba premade filtered 1D vector
 creation 3.701 time 0.0204
360.0
Test 7, Numba premade filtered 1D vector with for loop
 creation 0.1763 time 0.0224
360.0
Test 8, Numba premade 'smart' filtered 1D vector
 creation 0.1909 time 0.0216
360.0
Test 9, Python partial vectorized
 creation 0.0165 time 0.0086
360.0
"""

from numpy import round, array, arange
import numpy as np
from numba import njit
from itertools import product
from timeit import default_timer

N = 5

J = 3
s = 5
D = 25
dim = tuple(np.repeat([D+1, s+1], J))
size = np.prod(dim)

# Prework
sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

# Test 1, Itertools (product), all states
preliminary = 0  # Timing
start = default_timer()  # Timing
for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(dim)
    preliminary += default_timer() - start_2  # Timing
    it = np.nditer(memory, flags=['multi_index'])
    while not it.finished:
        multi_state = it.multi_index
        if(np.sum(multi_state[slice(J, J*2)]) <= s):
            memory[multi_state] = 1
        it.iternext()
print('Test 1, Itertools (product)\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory.ravel()*np.arange(size)) % 1000)


# Test 2, for loop premade x, s vector
preliminary = 0  # Timing
start = default_timer()  # Timing
for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(dim)
    s_states = array(list(product(arange(s+1), repeat=J)))
    x_states = array(list(product(arange(D+1), repeat=J)))
    preliminary += default_timer() - start_2  # Timing
    for s_state in s_states:
        if(np.sum(s_state) > s):
            continue
        for x_state in x_states:
            state = np.ravel([x_state, s_state])
            memory[tuple(state)] = 1
print('Test 2, for loop premade x, s vector\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory.ravel()*np.arange(size)) % 1000)

# Test 3, for loop premade x, s vector, prefilter invalid states
preliminary = 0  # Timing
start = default_timer()  # Timing
for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(dim)
    s_states = array(list(product(arange(s+1), repeat=J)))
    s_states = s_states[np.sum(s_states, axis=1) <= s]
    x_states = array(list(product(arange(D+1), repeat=J)))
    preliminary += default_timer() - start_2  # Timing
    for s_state in s_states:
        for x_state in x_states:
            state = np.ravel([x_state, s_state])
            memory[tuple(state)] = 1
print('Test 3, for loop premade x, s vector, prefilter invalid states\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory.ravel()*np.arange(size)) % 1000)

# Test 4, Numba premade x, s vector
preliminary = 0  # Timing
start = default_timer()  # Timing

@njit
def numba_2(memory, s_states, x_states, sizes, J):
    """Docstring."""
    for s_state in s_states:
        if(np.sum(s_state) > s):
            continue
        for x_state in x_states:
            state = np.sum(x_state*sizes[0:J] + s_state*sizes[J:J*2])
            memory[state] = 1

for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(size)
    s_states = array(list(product(arange(s+1), repeat=J)))
    x_states = array(list(product(arange(D+1), repeat=J)))
    preliminary += default_timer() - start_2  # Timing
    numba_2(memory, s_states, x_states, sizes, J)
print('Test 4, Numba premade x, s vector\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory*np.arange(size)) % 1000)

# Test 5, Numba premade x, s vector, prefilter invalid states
preliminary = 0  # Timing
start = default_timer()  # Timing

@njit
def numba_3(memory, s_states, x_states, sizes, J):
    """Docstring."""
    for s_state in s_states:
        for x_state in x_states:
            state = np.sum(x_state*sizes[0:J] + s_state*sizes[J:J*2])
            memory[state] = 1

for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(size)
    s_states = array(list(product(arange(s+1), repeat=J)))
    s_states = s_states[np.sum(s_states, axis=1) <= s]
    x_states = array(list(product(arange(D+1), repeat=J)))
    preliminary += default_timer() - start_2  # Timing
    numba_3(memory, s_states, x_states, sizes, J)
print('Test 5, Numba premade x,s, prefilter invalid states\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory*np.arange(size)) % 1000)

# Test 6, Numba premade filtered 1D vector
preliminary = 0  # Timing
start = default_timer()  # Timing

@njit
def numba_4(memory, states):
    """Docstring."""
    for state in states:
        memory[state] = 1

for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(size)
    states = array(list(product(arange(D+1), arange(s+1), repeat=J)))
    states = states[np.sum(states[:, arange(1, J*2, 2)], axis=1) <= s]
    states = states[:,np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])]
    states = np.sum(sizes * states, axis=1)
    preliminary += default_timer() - start_2  # Timing
    numba_4(memory, states)
print('Test 6, Numba premade filtered 1D vector\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory*np.arange(size)) % 1000)

# Test 7, Numba premade filtered 1D vector with for loop
preliminary = 0  # Timing
start = default_timer()  # Timing

@njit
def numba_prework(s_states, x_states, sizes):
    """Docstring."""
    states = np.zeros(len(s_states)*len(x_states), dtype=np.int64)
    counter = 0
    for s_state in s_states:
        for x_state in x_states:
            states[counter] = np.sum(x_state*sizes[0:J] +
                                     s_state*sizes[J:J*2])
            counter = counter + 1
    return states

@njit
def numba_5(memory, states):
    """Docstring."""
    for state in states:
        memory[state] = 1

for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(size)
    s_states = array(list(product(arange(s+1), repeat=J)))
    s_states = s_states[np.sum(s_states, axis=1) <= s]
    x_states = array(list(product(arange(D+1), repeat=J)))
    states = numba_prework(s_states, x_states, sizes)
    preliminary += default_timer() - start_2  # Timing
    numba_5(memory, states)
print('Test 7, Numba premade filtered 1D vector with for loop\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory*np.arange(size)) % 1000)

# Test 8, Numba premade 'smart' filtered 1D vector
preliminary = 0  # Timing
start = default_timer()  # Timing

@njit
def numba_prework_2(s_states, x_states, sizes):
    no_x_states = len(x_states)
    states = np.zeros(no_x_states*len(s_states), dtype=np.int64)
    counter = 0
    for s_state in s_states:
        states[counter:(counter+no_x_states)] = \
            np.sum(x_states*sizes[0:J] + s_state*sizes[J:J*2], axis=1)
        counter = counter + no_x_states
    return states

@njit
def numba_6(memory, states):
    """Docstring."""
    for state in states:
        memory[state] = 1

for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(size)
    s_states = array(list(product(arange(s+1), repeat=J)))
    s_states = s_states[np.sum(s_states, axis=1) <= s]
    x_states = array(list(product(arange(D+1), repeat=J)))
    states = numba_prework_2(s_states, x_states, sizes)
    preliminary += default_timer() - start_2  # Timing
    numba_6(memory, states)
print('Test 8, Numba premade \'smart\' filtered 1D vector\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
start = default_timer()
print(np.sum(memory*np.arange(size)) % 1000)

# Test 9, Python partial vectorized
preliminary = 0  # Timing
start = default_timer()  # Timing
for n in arange(N):
    start_2 = default_timer()  # Timing
    memory = np.zeros(dim)
    s_states = array(list(product(arange(s+1), repeat=J)))
    s_states = s_states[np.sum(s_states, axis=1) <= s]
    x_states = array(list(product(arange(D+1), repeat=J)))
    preliminary += default_timer() - start_2  # Timing
    for s_state in s_states:
        states = [slice(None)]*(J*2)
        states[slice(J,J*2)] = s_state
        memory[tuple(states)] = 1
print('Test 9, Python partial vectorized\n',
      'creation', round(preliminary/N, 4),
      'time', round((default_timer()-start-preliminary)/N, 4))
print(np.sum(memory.ravel()*np.arange(size)) % 1000)
