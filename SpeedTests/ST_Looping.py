"""
Looping, with J < s << D.

Take home,

numba is as fast as vectorization*.
numba is much faster than using python for loops.
Creating numba function takes no significant time (this holds even more for
calling numba functions)

*Need to have flatten array and work with ravel.
*Creating large matrices is slow! In that case numbda would be faster.
If a matrix manipulation can be done with large dimensions set to
':', '...' or 'None' (and only a small list of tuples left), then
vectorization can be a factor 2 - 10 faster than looping. Which is all
relative to the size of the dimension skipped and only is faster if the
skipped dimension is large enough.

Creating a sublarge matrix outside the loop can be efficient.

Of the current implementations, test 8 was the fastest, this is due to the
different approach (It creates a lot smaller help matrix, with only all
combinations of s (and not of s and x)), the mutation is smartly vectorized
(in a way that would not be possible in numba)

multi_index of nditer is a tuple
@jit(nopython=True) equivalent to @njit

when using numba jit, pass global variables as parameters
https://stackoverflow.com/questions/39371021/efficient-loop-over-numpy-array
# Supported functions
https://numba.pydata.org/numba-doc/dev/reference/pysupported.html

Test 1, for-loops
 time 6.1637
Test 2, looping with itertools
 time 5.9418
True
Test 3, looping with premade index vector
 time 5.4813
True
Test 4, numba jit and for loops
 time 0.2151
True
Test 5, all numba jit, flatten array, for loops
 time 0.2237
True
Test 6, numba jit, flatten array, looping with premade index vector
 time 0.1633
True
create looping vector 0.9375569000003452
 Do looping 0.1165
Test 7, all numba, flatten array, looping with premade index vector
 time 0.17
True
create looping vector 0.9358116000003065
 Do looping 0.1234
Test 8, Original Vectorized code
 time 0.006
True
Test 9, Vectorized
 time 1.5403
True
create all comb 18.175696100001915
 rearange array 0.2269678999991811
 delete elements from states 0.3042142000031163
 mutate matrix 12.081272699998408
Test 10, create function
 time 0.0
Test 10, numba jit, One function
 time 0.2147
True

"""

# import os
# os.chdir(r"D:\Documents\SURFdrive\VU\Promovendus"
#           r"\Time constraints in emergency departments\Code")
# import os
# os.chdir(r"C:\Users\jkn354\Documents\surfdrive\VU\Promovendus"
#           r"\Time constraints in emergency departments\Code")

from numpy import round
from numpy import arange
from numpy import array
import numpy as np
from itertools import product
from timeit import default_timer
from numba import njit
from Tools_FL import unravel_index
from Tools_FL import ravel_multi_index


N = 1

J = 3  # fixed at 2 for test 1.
s = 10
D = 50
dim = tuple(np.repeat([D+1, s+1], J))
size = np.prod(dim)

# Test 1, for-loops
start = default_timer()
for n in arange(N):
    matrix = np.zeros(dim)
    for x_1 in arange(D+1):
        for s_1 in arange(s+1):
            for x_2 in arange(D+1):
                for s_2 in arange(s+1):
                    multi_state = [x_1, x_2, s_1, s_2]
                    if(np.sum(multi_state[slice(J, J*2)]) > s):
                        matrix[tuple(multi_state)] = -1
print('Test 1, for-loops \n',
      'time', round((default_timer()-start)/N, 4))

# Test 2, looping with itertools
start = default_timer()
for n in arange(N):
    matrix2 = np.zeros(dim)
    it = np.nditer(matrix2, flags=['multi_index'])
    while not it.finished:
        multi_state = it.multi_index
        if(np.sum(multi_state[slice(J, J*2)]) > s):
            matrix2[multi_state] = -1
        it.iternext()
print('Test 2, looping with itertools \n',
      'time', round((default_timer()-start)/N, 4))
matrix = matrix2  # TODO
print(np.array_equal(matrix, matrix2))

# Test 3, looping with premade index vector
start = default_timer()
for n in arange(N):
    matrix3 = np.zeros(dim)
    states = array(list(product(arange(D+1), arange(s+1), repeat=J)))
    states = states[:, np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])]
    for state in states:
        if(np.sum(state[slice(J, J*2)]) > s):
            matrix3[tuple(state)] = -1
print('Test 3, looping with premade index vector \n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix3))

# Test 4, numba jit, flatten array, for loops
start = default_timer()


@njit
def numba_jit4(J, s, D, dim, size):
    """Docstring."""
    matrix = np.zeros(size, dtype=np.int64)
    for index in arange(size):
        multi_state = unravel_index(index, dim)
        if(np.sum(multi_state[J:J*2]) > s):
            matrix[index] = -1
    return matrix


for n in arange(N):
    matrix4 = numba_jit4(J, s, D, dim, size)
print('Test 4, numba jit and for loops \n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix4.reshape(dim)))

# Test 5, all numba jit, flatten array, for loops
start = default_timer()


@njit
def numba_jit5(J, s, D, dim, size):
    """Docstring."""
    matrix = np.zeros(size, dtype=np.int64)
    for index in arange(size):
        multi_state = unravel_index(index, dim)
        if(np.sum(multi_state[J:J*2]) > s):
            matrix[index] = -1
    return matrix


@njit
def f(J, s, D, dim, size, N):
    """Docstring."""
    for n in arange(N):
        matrix4 = numba_jit5(J, s, D, dim, size)
    return matrix4


matrix5 = f(J, s, D, dim, size, N)
print('Test 5, all numba jit, flatten array, for loops \n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix5.reshape(dim)))


# Test 6, numba jit, flatten array, looping with premade index vector
start = default_timer()
time1 = 0


@njit
def numba_jit6(J, s, D, dim, size, states):
    """Docstring."""
    matrix = -np.ones(size, dtype=np.int64)
    for multi_state in states:
        index = ravel_multi_index(multi_state, dim)
        matrix[index] = 0
    return matrix


_start = default_timer()
states = array(list(product(arange(D+1), arange(s+1), repeat=J)))
states = states[:, np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])]
states = states[np.sum(states[:, J:J*2], axis=1) <= s]
time1 += default_timer() - _start
for n in arange(N):
    matrix6 = numba_jit6(J, s, D, dim, size, states)
print('Test 6, numba jit, flatten array, looping with premade index vector\n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix6.reshape(dim)))
print('create looping vector', time1, '\n',
      'Do looping', round((default_timer()-time1-start)/N, 4))

# Test7, all numba, flatten array, looping with premade index vector
start = default_timer()
time1 = 0


@njit
def numba_jit7(J, s, D, dim, size, states):
    """Docstring."""
    matrix = -np.ones(size, dtype=np.int64)
    for multi_state in states:
        index = ravel_multi_index(multi_state, dim)
        matrix[index] = 0
    return matrix


@njit
def f(J, s, D, dim, size, states, N):
    """Docstring."""
    for n in arange(N):
        matrix7 = numba_jit7(J, s, D, dim, size, states)
    return matrix7


_start = default_timer()
states = array(list(product(arange(D+1), arange(s+1), repeat=J)))
states = states[:, np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])]
states = states[np.sum(states[:, J:J*2], axis=1) <= s]
time1 += default_timer() - _start
matrix7 = f(J, s, D, dim, size, states, N)
print('Test 7, all numba, flatten array, looping with premade index vector\n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix7.reshape(dim)))
print('create looping vector', time1, '\n',
      'Do looping', round((default_timer()-time1-start)/N, 4))

# Test 8, Original Vectorized code
start = default_timer()
for n in arange(N):
    matrix8 = np.zeros(dim)
    s_states = product(arange(s+1), repeat=J)
    s_states = np.array(list(s_states))
    s_states = s_states[np.sum(s_states, axis=1) > s]
    for s_state in s_states:
        multi_state = np.array([[slice(None)]*J, s_state]).reshape(-1)
        matrix8[tuple(multi_state)] = -1
print('Test 8, Original Vectorized code\n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix8))

# Test 9, Vectorized
start = default_timer()
time1 = 0
time2 = 0
time3 = 0
time4 = 0
for n in arange(N):
    matrix9 = np.zeros(dim)

    _start = default_timer()
    states = array(list(product(arange(D+1), arange(s+1), repeat=J)))
    time1 += default_timer() - _start

    _start = default_timer()
    states = states[:, np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])]
    time2 += default_timer() - _start

    _start = default_timer()
    states = states[np.sum(states[:, J:J*2], axis=1) > s]
    time3 += default_timer() - _start

    _start = default_timer()
    matrix9[tuple(zip(*states))] = -1
    time4 += default_timer() - _start
print('Test 9, Vectorized\n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix9))
print('create all comb', time1, '\n',
      'rearange array', time2, '\n',
      'delete elements from states', time3, '\n',
      'mutate matrix', time4)

# Test 10, Failed Vectorize
# start = default_timer()
# for n in arange(N):
#     matrix10 = np.zeros(dim)
#     s_states = product(arange(s+1), repeat=J)
#     s_states = np.array(list(s_states))
#     s_states = s_states[np.sum(s_states, axis=1) > s]
#     d_states = np.repeat([None]*J, len(s_states))
#     d_states = d_states.reshape((len(s_states), J))
#     matrix10[tuple(zip(*np.append(d_states, s_states, axis=1)))] = -1
# print('Test 10, Original Vectorized code\n',
#       'time', round((default_timer()-start)/N, 4))
# print(np.array_equal(matrix, matrix10))

# Test 10, numba jit, One function
start = default_timer()


@njit
def numba_jit10(J, s, D, dim, size, N):
    """Docstring."""
    for n in arange(N):
        matrix = np.zeros(size, dtype=np.int64)
        for index in arange(size):
            sizes = np.zeros(len(dim), dtype=np.int64)
            multi_state = np.zeros(len(dim), dtype=np.int64)
            sizes[-1] = 1
            for i in range(len(dim) - 2, -1, -1):
                sizes[i] = sizes[i + 1] * dim[i + 1]
            remainder = index
            for i in range(len(dim)):
                multi_state[i] = remainder // sizes[i]
                remainder %= sizes[i]
            if(np.sum(multi_state[J:J*2]) > s):
                matrix[index] = -1
    return matrix


print('Test 10, create function \n',
      'time', round((default_timer()-start)/N, 4))
start = default_timer()

matrix10 = numba_jit10(J, s, D, dim, size, N)
print('Test 10, numba jit, One function \n',
      'time', round((default_timer()-start)/N, 4))
print(np.array_equal(matrix, matrix10.reshape(dim)))
