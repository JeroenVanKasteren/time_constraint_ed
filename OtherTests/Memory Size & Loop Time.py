"""
Take home.

Stay away from np.empty!

@author: Jeroen
"""

from numpy import array
import numpy as np
from itertools import product
from sys import getsizeof
from numpy import arange
import timeit

# Memory size of matrices
D = 10
s = 5
J = 3
dim = (D+1, s+1)*J
print(((D+1)*(s+1))**J * 8 / 10**9)  # memory size in GB

policy = np.zeros(dim)
print(getsizeof(policy)/10**9)  # GB
print((np.prod(policy.shape) * 8) / 10**9)  # memory size in GB

policy_ = np.empty(dim)
print(getsizeof(policy_)/10**9)  # Does not matter in size

policy_int = np.zeros(dim, dtype=int)
print(getsizeof(policy)/10**9)

matrix = np.random.rand(D, s, D, s, D, s)
print(getsizeof(policy)/10**9)

# Looping
D = 1
s = 2
J = 2
dim = (D+1, s+1)*J
memory = np.arange(((D+1)*(s+1))**J).reshape(dim)
mask = np.random.randint(0, 2, dim)

for (index, val), m in zip(np.ndenumerate(memory), mask):
    print(index, val, m)

pointer = 0
# for s_state in s_states:
s_state = s_states[0]
w_m = np.zeros([J]+[D+1]*J)
v_m = np.zeros([D+1]*J)
for i in range(J):
    for state in D_1_states:
        pointer += 1
        state = list(state)
        state.insert(i, slice(None))
        w_m[tuple([i]+state)] = np.random.randint(10)
v_m[tuple([slice(None)]*J)] = 3  # V[multi_state with x=[slice(None)]]
# Penalty
for i in range(J):
    state = [slice(None)]*(J-1)
    state.insert(i, D)
    v_m[tuple(state)] = np.inf
v_m[tuple([D]*J)]
print(w_m)
print(v_m)
# get slice of W(x,s) which we can fill for given s_state
# np.array([[slice(None)]*J, s_state]).T.reshape(-1)
np.min(np.concatenate((w_m, [v_m]), 0), 0)

index = (2, 0, 3, 1, 4, 0)
J = 3
states = np.array(index).reshape(-1, 2)
for i, state in enumerate(states):
    print(i, state)

B = np.empty(shape=[3, 1, 4, 2])
it = np.nditer(B, flags=['multi_index'])
while not it.finished:
    print(it[0], it.multi_index)
    print(np.array(it.multi_index).reshape(-1, 2))
    B[it.multi_index] = -1
    it.iternext()

# Break
for i in arange(3):
    for j in arange(3):
        if(j == 1):
            break
        print(i, j)

# Continue
for i in arange(3):
    for j in arange(3):
        if(j == 1):
            continue
        print(i, j)
