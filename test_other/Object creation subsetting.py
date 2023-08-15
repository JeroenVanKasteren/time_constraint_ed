"""
Take home.

How to create a list
Tuples are immutable

Use arange (over range and xrange) for subsetting and list/array creation
Source:
https://stackoverflow.com/questions/10698858/built-in-range-or-numpy-arange-which-is-more-efficient
tuples must be subsetted with slice()

@author: Jeroen
"""

from numpy import array, arange
from itertools import product
import numpy as np

# get all x_i or s_i from multi_state tuple
D = 4
s = 3
J = 2
dim = (D+1, s+1)*J

x = [1]*J
s = [2]*J
np.array([x, s]).T.reshape(-1)

# List/array creation
array([0, 1, 2, 3])
array(arange(3+1))

# Creating a list, does make a copy
tup = (1, 2, 3, 4)
tup[2]
# tup[2] = 4  # ERROR
next_state = list(tup)
next_state[2] = 0
print(next_state, tup)  # Not linked anymore

# Tuple (dim) subsetting
# dim[arange(0, J*2, 2)]  # ERROR

array(dim)[arange(0, J*2, 2)]  # x_i
array(dim)[arange(1, J*2, 2)]  # s_i

dim[slice(0, J*2, 2)]  # x_i
dim[slice(1, J*2, 2)]  # s_i

D = 1
s = 3
J = 2
# Create list of all states, x1, s1, x2, s2 ...
states = list(product(arange(D+1), arange(s+1), repeat=J))
# Create list of all valid states
s_states = array(list(product(arange(s+1), repeat=J)))
s_states = s_states[np.sum(s_states, axis=1) <= s]
D_states = list(product(arange(D+1), repeat=J))
# x1, x2, ..., s1, s2
states_ = array(list(product(D_states, s_states))).reshape(-1, 2*J)
# x1, s1, x2, s2, ...
x_index = arange(J)
s_index = arange(J, J*2)
tmp = array([x_index, s_index]).T.reshape(-1)
states = states_[:, tmp]

policy = np.zeros(dim)
masking_x = list(dim)
for i in range(J):
    masking_x[i*2] = range(D+1)


dim = array([(3, 2) for j in range(2)]).flatten()
A = arange((3*2)**2).reshape(dim)
x = [1,0]
s = [1,1]
state = np.stack((x, s), axis=1).flatten()
state[0] = [1,2,3]  # Error!
state = list(np.stack((x, s), axis=1).flatten())
state[0] = range(1,3)
A[tuple(state)]
state[0] = range(2)
A[tuple(state)]
state[1] = range(2)
A[tuple(state)]

x = []
x.append([0,[0,0,0]]); x
x.append([1,1,1,1]); x

s = 3; D = 20; state = 5
x = np.array(range(-s, D+1))
print(x, x[0], x[state+s], x[D+s])

A = np.indices((D+1, D+1))  # x=A[0], y=A[1]
A[0, 1:D+1, 0]
A[(0, arange(1, D+1), 0)]  # is the same as previous line

# ------------------ (i, x, s) ----------------
D = 10
S = 4
J = 2
# (x, s)
tuple(np.repeat([D+1, S+1], J))
# (i, x, s)
np.append(J, np.repeat([D+1, S+1], J))  # Only for 2 elements

i = 1
x_state = [2, 3]
s_state = [4, 5]
np.concatenate(([i], x_state, s_state), axis=0)  # Slightly longer code, but for >2 elements

# --------------------- Subsetting ------------------
# get all x_i or s_i from multi_state tuple
D = 2
s = 1
J = 2
dim = (D+1, s+1)*J

# Creating a list, does make a copy
tup = (1, 2, 3, 4)
next_state = list(tup)
next_state[2] = 0
print(next_state, tup)  # Not linked anymore

np.array(dim)[range(0, J*2, 2)]  # x_i
np.array(dim)[arange(0, J*2, 2)]  # x_i
dim[slice(0, J*2, 2)]  # x_i, need to use slice to subset tuple

np.array(dim)[arange(1, J*2, 2)]  # s_i

dim[slice(1, J*2, 2)]  # s_i
policy = np.zeros(dim)
masking_x = list(dim)
for i in arange(J):
    masking_x[i*2] = arange(D+1)

A = np.arange((D+1)**2 * (s+1)**2).reshape([D+1, s+1, D+1, s+1])
multi_state = [1, 0, 1, 1]
A[tuple(multi_state)]
A[multi_state]  # Deprecated, unexpected output!
A[np.array(multi_state)]  # Deprecated, unexpected output!


np.unravel_index([multi_state], shape=dim)

multi_state = [arange(D-1), 1, 0, 0]
A[tuple(multi_state)]  # Works

multi_state = [arange(D+1), 1, arange(D+1), 0]
A[tuple(multi_state)]  # Works, note that it is element-wise

# The following does not work, ambiguous due to element-wise interpretation
multi_state = [arange(D+1), 1, arange(D+1), arange(s)]
# A[tuple(multi_state)]  # Error

A[arange(D+1), 1, :, :]  # This does work, but not for a tuple
multi_state = [arange(D+1), 1, slice(None), slice(None)]
A[tuple(multi_state)]  # This works

multi_state = [slice(D+1), 1, slice(None), slice(None)]
A[tuple(multi_state)]  # This works

multi_state = [arange(3, -1, -1), 1, slice(None), slice(None)]
A[tuple(multi_state)]  # This works

multi_state = [slice(D, None, -1), 1, slice(None), slice(None)]
A[tuple(multi_state)]  # This works