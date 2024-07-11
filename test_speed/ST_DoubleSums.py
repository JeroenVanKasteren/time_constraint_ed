"""
Caclulating V(x) for x = -s, ..., 0

Take Home
Vectorized makes everything faster, even one multiplication

Test 3.x shows how masking is the fastes way to subset
Test 4.x shows that masking is a fast way to do elementwise matrix arithmatics.

With large s (=100)

Test 1, all for loops, inside multiplication
time 0.4467 mean V 84.5878

Test 2, all for loops, outside multiplication
time 0.4169 mean V 84.5878

Test 3, using recursion, vectorized per i, inside multiplication
time 0.0026 mean V 84.5878

Test 4, using recursion, vectorized per i, outside multiplication
time 0.0027 mean V 84.5878

Test 5, all multiplications vectorized (even doing twice as many)
time 0.0014 mean V 84.5878

Test 6, vectorized with mask
time 0.0013 mean V 84.5878

Test 7, vectorized with mask and cumsum
time 0.0006 mean V 84.5878

Test 3.1, set above diagonal to zero, tril_indices
8.3 ms ± 314 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Test 3.2, set above diagonal to zero, tri
8.29 ms ± 366 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Test 3.3, set above diagonal to zero, mask
1.09 ms ± 20.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Test 4.1, apply to whole matrix
time 0.0006 mean M 0.0878

Test 4.2, using mask
time 0.0005 mean M 0.0878

Test 4.3, using mask, deleting columns
time 0.0005 mean M 0.087
"""

import numpy as np
from scipy.special import factorial as fac
import timeit

lambda_ = 1
s = 100
mu = (lambda_*0.95)/s
g = 1  # just placeholder, not real g
N = 100  # Repeat calculation

# Test 1, all for loops, inside multiplication
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    for x in range(1, s+1):  # V(-s) = V[0] = 0
        _V = 0
        for i in range(1, x+1):
            for j in range(i-1+1):
                _V += g / lambda_ * fac(i - 1) / fac(i - j - 1) * (mu/lambda_)**j
        V[x] = _V
print('Test 1, all for loops, inside multiplication')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 2, all for loops, outside multiplication
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    for x in range(1, s+1):  # V(-s) = V[0] = 0
        _V = 0
        for i in range(1, x+1):
            for j in range(i-1+1):
                _V += fac(i - 1) / fac(i - j - 1) * (mu/lambda_)**j
        V[x] = _V * g / lambda_
print('Test 2, all for loops, outside multiplication')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 3, using recursion, vectorized per i, inside multiplication
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    for i in range(1, s+1):  # V(-s) = V[0] = 0
        j = np.array(range(i))
        V[i] = V[i-1] + sum(g/lambda_ * (fac(i-1)/fac(i-j-1) * (mu/lambda_)**j))
print('Test 3, using recursion, vectorized per i, inside multiplication')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 4, using recursion, vectorized per i, outside multiplication
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    for i in range(1, s+1):  # V(-s) = V[0] = 0
        j = np.array(range(i))
        V[i] = V[i-1] + sum((fac(i-1)/fac(i-j-1) * (mu/lambda_)**j))
    V = g/lambda_*V
print('Test 4, using recursion, vectorized per i, outside multiplication')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 5, all multiplications vectorized (even doing twice as many)
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    # [[i index matrix] [j index matrix]]
    A = np.delete(np.indices((s+1, s+1)), 0, 1)
    A = np.delete(A, s, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        M = fac(A[0] - 1) / fac(A[0] - A[1] - 1) * (mu/lambda_)**A[1]
    for i in range(1, s+1):  # V(-s) = V[0] = 0
        V[i] = V[i-1] + sum(M[i-1, 0:i-1+1])
    V = g/lambda_*V
print('Test 5, all multiplications vectorized (even doing twice as many)')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 6, vectorized with mask
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    # [[i index matrix] [j index matrix]]
    A = np.delete(np.indices((s+1, s+1)), 0, 1)
    A = np.delete(A, s, 2)
    M = np.zeros([s, s])
    # Row index >= Column index, Lower Triangular
    mask = np.arange(A[0].shape[0])[:, None] >= np.arange(A[0].shape[1])
    M[mask] = fac(A[0, mask] - 1) / fac(A[0, mask] - A[1, mask] - 1) * (mu/lambda_)**A[1, mask]
    for i in range(1, s+1):  # V(-s) = V[0] = 0
        V[i] = V[i-1] + sum(M[i-1, 0:i-1+1])
    V = g/lambda_*V
print('Test 6, vectorized with mask')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 7, vectorized with mask and cumsum
start = timeit.default_timer()
for n in range(N):
    V = np.zeros(s+1)
    # [[i index matrix] [j index matrix]]
    A = np.delete(np.indices((s+1, s+1)), 0, 1)
    A = np.delete(A, s, 2)
    M = np.zeros([s, s])
    # Row index >= Column index, Lower Triangular
    mask = np.arange(A[0].shape[0])[:, None] >= np.arange(A[0].shape[1])
    M[mask] = fac(A[0, mask] - 1) / fac(A[0, mask] - A[1, mask] - 1) * (mu/lambda_)**A[1, mask]
    V[1:] = np.cumsum(np.sum(M, axis=1))
    V = g/lambda_*V
print('Test 7, vectorized with mask and cumsum')
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean V', round(np.mean(V), 4), '\n')

# Test 2.1, factorial whole matrix
N = 1000
s = 4
A = np.delete(np.indices((s+1, s+1)), 0, 1)
A = np.delete(A, s, 2)
B = A
start = timeit.default_timer()
for n in range(N):
    B = fac(A[0] - 1)
print('Test 2.1, factorial whole matrix')
print('time', round(timeit.default_timer()-start, 4),
      'mean A', round(np.mean(B[:, 0]), 4), '\n')

# Test 2.2, factorial using zeros
A = np.delete(np.indices((s+1, s+1)), 0, 1)
A[0] = np.tril(A[0], k=0)
start = timeit.default_timer()
for n in range(N):
    B = fac(A[0] - 1)
print('Test 2.2, factorial using zeros')
print('time', round(timeit.default_timer()-start, 4),
      'mean A', round(np.mean(B[:, 0]), 2), '\n')

# Test 2.3, factorial using mask
A = np.delete(np.indices((s+1, s+1)), 0, 1)
start = timeit.default_timer()
for n in range(N):
    B = fac(A[0][np.tril_indices(4)] - 1)
print('Test 2.3, factorial using mask')
print('time', round(timeit.default_timer()-start, 4),
      'mean A', round(np.mean(B[:, 0]), 4), '\n')

# ------------------------------ 3 -------------------------------- #
# stackoverflow.com/questions/41070766/how-to-multiply-outer-in-numpy-while-assuming-0-infinity-0
# stackoverflow.com/questions/23839688/how-to-fill-upper-triangle-of-numpy-array-with-zeros-in-place
i = 10**3
j = i
print('Test 3.1, set above diagonal to zero, tril_indices')
M = np.arange(i*j).reshape(i, j) + 1
# %timeit M[np.triu_indices(M.shape[0], 1)] = 0

print('\nTest 3.2, set above diagonal to zero, tri')
M = np.arange(i*j).reshape(i, j) + 1
# %timeit B = M * np.tri(*M.shape)

print('\nTest 3.2, set above diagonal to zero, mask')
M = np.arange(i*j).reshape(i, j) + 1
# %timeit M[np.arange(M.shape[0])[:, None] < np.arange(M.shape[1])] = 0

print('\nTest 3.3, set above diagonal to zero, mask and np.where')
M = np.arange(i*j).reshape(i, j) + 1
# %timeit B = np.where(np.arange(M.shape[0])[:, None] < np.arange(M.shape[1]), 0, M)

# ------------------------------ 4 -------------------------------- #
N = 1000
s = 100
print('Test 4.1, apply to whole matrix')
start = timeit.default_timer()
A = np.delete(np.indices((s+1, s+1)), 0, 1)
A = np.delete(A, s, 2)
for n in range(N):
    with np.errstate(divide='ignore', invalid='ignore'):
        M = fac(A[0] - 1) / fac(A[0] - A[1] - 1) * (mu/lambda_)**A[1]
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean M', round(np.mean(M[s-1, :]), 4), '\n')

print('Test 4.2, using mask')
start = timeit.default_timer()
A = np.delete(np.indices((s+1, s+1)), 0, 1)
M = A[0].copy().astype(float)
for n in range(N):
    mask = np.arange(A[0].shape[0])[:, None] >= np.arange(A[0].shape[1])
    M[mask] = fac(A[0, mask] - 1) / fac(A[0, mask] - A[1, mask] - 1) * \
        (mu/lambda_)**A[1, mask]
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean M', round(np.mean(M[s-1, :-1]), 4), '\n')

print('Test 4.3, using mask, deleting columns')
start = timeit.default_timer()
A = np.delete(np.indices((s+1, s+1)), 0, 1)
A = np.delete(A, s, 2)
M = A[0].copy().astype(float)
for n in range(N):
    mask = np.arange(A[0].shape[0])[:, None] >= np.arange(A[0].shape[1])
    M[mask] = fac(A[0, mask] - 1) / fac(A[0, mask] - A[1, mask] - 1) *\
        (mu/lambda_)**A[1, mask]
print('time', round((timeit.default_timer()-start)/N, 4),
      'mean M', round(np.mean(M[s-1, :]), 4), '\n')
