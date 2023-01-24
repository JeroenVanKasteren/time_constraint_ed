"""
Testing Transition Probabilities.

Test 1, one class
time 0.0254 test 13

Test 2, one class vectorized
time 0.0017 test 13

Mutliple classes (Do not compare time)
time 0.0001 Visual test, compare with test 1 or 2

Precalculate P_xy.
Jump from x>0 to y. For index convenience P_0y=0
"""

import numpy as np
import timeit

np.set_printoptions(precision=3)

N = 100  # Repeat calculation
D = 300

lambda_ = 1
gamma = 5

# Test 1, one class
start = timeit.default_timer()
P_xy = np.zeros((D+1, D+1))
for n in range(N):
    for x in range(D+1):
        for y in range(1, x+1):
            P_xy[(x, y)] = (gamma / (lambda_ + gamma))**(x-y) * \
                lambda_ / (lambda_ + gamma)
    P_xy[1:D+1, 0] = (gamma / (lambda_ + gamma))**np.array(range(1, D+1))
print('Test 1, one class')
print('time', round((timeit.default_timer()-start)/N, 4),
      'test', round(sum(np.sum(P_xy, axis=1) == 1), 4), '\n')

# Test 2, one class vectorized
start = timeit.default_timer()
P_xy = np.zeros((D+1, D+1))  # for class i in J, p_{x,y}
for n in range(N):
    A = np.indices((D+1, D+1))  # x=A[0], y=A[1]
    mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
    P_xy[1:, 1:][mask_tril] = (gamma / (lambda_ + gamma)) ** \
        (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
        lambda_ / (lambda_ + gamma)
    P_xy[1:, 0] = (gamma / (lambda_ + gamma)) ** A[0, 1:, 0]
print('Test 1, one class')
print('time', round((timeit.default_timer()-start)/N, 4),
      'test', round(sum(np.sum(P_xy, axis=1) == 1), 4), '\n')

J = 3
lambda_V = np.array([1, 1.5, 0.5])
D = 4

# Mutliple classes
start = timeit.default_timer()
P_xy = np.zeros((J, D+1, D+1))
for n in range(N):
    A = np.indices((D+1, D+1))  # x=A[0], y=A[1]
    mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
    for i in range(J):
        P_xy[i, 1:, 1:][mask_tril] = (gamma / (lambda_V[i] + gamma)) ** \
            (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
            lambda_V[i] / (lambda_V[i] + gamma)
        P_xy[i, 1:, 0] = (gamma / (lambda_V[i] + gamma)) ** A[0, 1:, 0]
print('Test 1, Mutliple classes')
print('time', round((timeit.default_timer()-start)/N, 4),
      'Visual test, compare with test 1 or 2 \n')
np.sum(P_xy, axis=(2))