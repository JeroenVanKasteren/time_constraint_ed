"""
Take home,

Numba versus vectorized code: depends...
n=1000000 --> Numba beats vectorized code 10 - 100x
for loops are faster than vectorization!
Test 1, vectorized
 time 0.2037
Test 2, numba
 time 0.0341
Test 2, numba
 time 0.0467
Test 2, numba
 time 0.0051
Test 2, numba
 time 0.005

n=10000 --> Numba is 2x-10x slower
Test 1, vectorized
 time 0.001
Test 2, numba
 time 0.0055
Test 2, numba
 time 0.0055
Test 2, numba
 time 0.0076
Test 2, numba
 time 0.0054
"""

from numpy import round
from numpy import arange
import numpy as np
from timeit import default_timer
from numba import njit

N = 30
n = 1000000
J = 5
matrix = np.zeros(n)

# Test 1, for-loops
start = default_timer()
for n in arange(N):
    for i in arange(J):
        matrix = np.cos(matrix+1) ** 2 + np.sin(matrix+1) ** 2
print('Test 1, vectorized \n',
      'time', round((default_timer()-start)/N, 4))

# Test 2, numba
@njit(fastmath=True)
def test_2(matrix, J):
    """Docstring."""
    for i in arange(J):
        matrix = np.cos(matrix+1) ** 2 + np.sin(matrix+1) ** 2
    return matrix

start = default_timer()
for n in arange(N):
    matrix = test_2(matrix, J)
print('Test 2, numba \n',
      'time', round((default_timer()-start)/N, 4))

# Test 2, numba
@njit
def test_2(matrix, J):
    """Docstring."""
    for i in arange(J):
        matrix = np.cos(matrix+1) ** 2 + np.sin(matrix+1) ** 2
    return matrix

start = default_timer()
for n in arange(N):
    matrix = test_2(matrix, J)
print('Test 2, numba \n',
      'time', round((default_timer()-start)/N, 4))

# Test 3, numba
@njit(fastmath=True)
def test_3(matrix, J, n):
    """Docstring."""
    for i in arange(J):
        for x in arange(n):
            matrix[x] = np.cos(matrix[x]+1) ** 2 + np.sin(matrix[x]+1) ** 2
    return matrix

start = default_timer()
for n in arange(N):
    matrix = test_3(matrix, J, n)
print('Test 2, numba \n',
      'time', round((default_timer()-start)/N, 4))

# Test 3, numba
@njit
def test_3(matrix, J, n):
    """Docstring."""
    for i in arange(J):
        for x in arange(n):
            matrix[x] = np.cos(matrix[x]+1) ** 2 + np.sin(matrix[x]+1) ** 2
    return matrix

start = default_timer()
for n in arange(N):
    matrix = test_3(matrix, J, n)
print('Test 2, numba \n',
      'time', round((default_timer()-start)/N, 4))

