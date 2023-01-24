"""
Take home.

For variable multidimensional matrices, tuples are used (as explains below)
https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
However, this is not possible in numba

# Supported functions
https://numba.pydata.org/numba-doc/dev/reference/pysupported.html
# Supported Numpy functions
https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

Numba cannot work with the object 'self'. Therefore, make all numba functions
@staticmethod and give all variables as parameters to a njit function.

Numba classes are impractical for this project as it cannot do vectorized
operations on multidimensional arrays, which are faster than numba in some
occasions.
Numba classes
https://numba.pydata.org/numba-doc/latest/user/jitclass.html?highlight=class#basic-usage
"""

# ---------------------------- TUPLES CREATION -----------------------------
# import os
# os.chdir(r"D:\Documents\SURFdrive\VU\Promovendus"
#           r"\Time constraints in emergency departments\Code")
# os.chdir(r"C:\Users\jkn354\surfdrive\VU\Promovendus"
#           r"\Time constraints in emergency departments\Code")

from numpy import arange, array, prod, dot, zeros
import numpy as np
from itertools import product
from numba import njit
# from Tools_FL import create_tuple_creator, unravel_index, ravel_multi_index

# get all x_i or s_i from multi_state tuple
D = 2
s = 1
J = 2


@njit
def f1(D, s, J):
    """Create tuples."""
    dim = (D+1, s+1)*J  # Not possible to create tuples like this
    return dim


@njit
def f2(D, s, J):
    """Create tuples."""
    dim = np.repeat((D+1, s+1), J)
    return dim


# nb_tuple = create_tuple_creator(J*2)


@njit
def f3(D, s, J):
    """Create tuples."""
    dim = np.repeat((D+1, s+1), J)
    return nb_tuple(dim)


f1(D, s, J)  # Error
f2(D, s, J)  # Returns array, not tuple
f3(D, s, J)  # Returns tuple



@njit
def f4(D, s, J):
    """Create matrix."""
    dim = np.repeat([D+1, s+1], J)
    return np.zeros(dim)


@njit
def f5(D, s, J):
    """Create matrix."""
    dim = np.repeat((D+1, s+1), J)
    return np.zeros(dim)


f4(D, s, J)  # Error (would work in python)
f5(D, s, J)  # Error (would work in python)

# -----------------------------------------------------------------------
# ---------------------------- MANIPULATION -----------------------------

@njit
def f6(D, s, J):
    """Tuple manipulation."""
    # Creating a list, does make a copy
    tup = (1, 2, 3, 4)
    next_state = list(tup)
    next_state[2] = 0
    print(next_state, tup)  # Not linked anymore


f6(D, s, J)  # Works

# -----------------------------------------------------------------------
# ---------------------------- SUBSETTING -------------------------------

@njit
def f7(D, s, J):
    """Slicing array."""
    dim = np.repeat((D+1, s+1), J)
    print(np.array(dim)[range(0, J*2, 2)])  # x_i
    print(np.array(dim)[arange(1, J*2, 2)])  # s_i


@njit
def f8(D, s, J):
    """Slicing array."""
    dim = np.repeat((D+1, s+1), J)
    print(dim[slice(0, J)])  # x_i, need to use slice to subset tuple
    print(dim[slice(J, J*2)])  # s_i


f7(D, s, J)  # Error (would work in python)
f8(D, s, J)  # Works


@njit
def f9(D, s, J):
    """Slicing tuple, python needs slice to subset tuple."""
    dim = nb_tuple(np.repeat((D+1, s+1), J))
    print(dim[slice(0, J)])  # x_i
    print(dim[slice(J, J*2)])  # s_i


@njit
def f10(D, s, J):
    """Slicing tuple, python needs slice to subset tuple."""
    dim = nb_tuple(np.repeat((D+1, s+1), J))
    print(dim[0], dim[1])  # x_i
    print(dim[2], dim[3])  # s_i


f9(D, s, J)  # Error (would work in python)
f10(D, s, J)  # Works

@njit
def f11(A, multi_state):
    """Variable indices with a tuple."""
    print(A[tuple(multi_state)])


@njit
def f12(A, multi_state):
    """Variable indices with a array."""
    print(A[multi_state])  # Deprecated, unexpected output!


@njit
def f13(A, multi_state):
    """Variable indices with a array."""
    print(A[np.array(multi_state)])  # Deprecated, unexpected output!


A = np.arange((D+1)**2 * (s+1)**2).reshape([D+1, D+1, s+1, s+1])
multi_state = [1, 0, 1, 1]
f11(A, multi_state)  # Error
f12(A, multi_state)  # Error
f13(A, multi_state)  # Error

# -----------------------------------------------------------------------
# ------------------ OPERATIONS SUBSETTED FLAT ARRAYS -------------------

@njit
def subset(A, states):
    """Docstring."""
    print(A[states])


@njit
def product_(A, states):
    """Docstring."""
    print(prod(A[states]))


@njit
def product_2(A, states):
    """Docstring."""
    print(A[states]*A[states])


@njit
def sum_(A, states):
    """Docstring."""
    print(A[states]+A[states])


@njit
def sumproduct(A, states):
    """Docstring."""
    print(dot(A[states],(A[states])))

A = np.arange((D+1)**2 * (s+1)**2).reshape([D+1, D+1, s+1, s+1])
A = A.flatten()
states = array([2, 3, 8, 9])
subset(A, states)
product_(A, states)
product_2(A, states)
sum_(A, states)
sumproduct(A, states)  # Error
A = A.astype(np.float32)
sumproduct(A, states)

# -----------------------------------------------------------------------
# --------------------- GLOBAL AND LOCAL MUTATIONS ----------------------

@njit
def global_local_test(states):
    """Docstring."""
    states[0] = 5
    newstates = states
    newstates[0] = -1
    print(newstates)
    newstates = states.copy()
    newstates[1] = 10
    print(newstates)

states = array([2, 3, 8, 9])
global_local_test(states)
print(states)

@njit
def local_reshape(states):
    """Docstring."""
    states = states.reshape(4)
    print(states)

states = array([[2, 3],[8, 9]])
local_reshape(states)
print(states)

global_constant = 2
global_variable = 3

@njit
def global_test():
    """Docstring."""
    return global_constant + global_variable

@njit
def global_test_2(local_variable):
    """Docstring."""
    return global_constant + local_variable

print(global_test())
print(global_test_2(global_variable))
global_variable = 4
print(global_test())
print(global_test_2(global_variable))

# -----------------------------------------------------------------------
# -------------------------- Looping in Numba ---------------------------

J = 2
s = 2
D = 3
dim = tuple(np.repeat([D+1, s+1], J))
size = np.prod(dim)

@njit
def states_f(s_states, x_states, J, order):
    for s_state in s_states:
        if(np.sum(s_state) > s):
            continue
        for x_state in x_states:
            lst = [x_state, s_state]
            flattend = array([x for sublst in lst for x in sublst])
            tmp = flattend[order]
            print(tmp)

order = np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])
s_states = array(list(product(arange(s+1), repeat=J)))
x_states = array(list(product(arange(D+1), repeat=J)))
states_f(s_states, x_states, J, order)


# The following statements do not work in Numba
# As creating an array from an array(s) is not possible
# np.concatenate(tmp)
# index = np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])
# array(tmp).reshape(-1)

s_states = array(list(product(arange(s+1), repeat=J)))
s_states = s_states[np.sum(s_states, axis=1) <= s]
x_states = array(list(product(arange(D+1), repeat=J)))

# -----------------------------------------------------------------------
# ----------------------------- RAVELLING -------------------------------

J = 2
s = 2
D = 3
dim = np.repeat([D+1, s+1], J)
size = np.prod(dim)

sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]
sizes

@njit
def numba_mesh(J):
    """Docstring."""
    x = array([5, 20])
    y = array([9, 10, 11, 12])
    print(np.repeat(x, J).reshape((-1, J)))
    print(np.repeat(x, J).reshape((-1, J)).T.ravel())
    # print(np.repeat(x,2).reshape((-1, 2)).T.reshape(-1))
    print(np.repeat(x, J))
    # print(np.indices(x))  # Not implemented
    # print(np.meshgrid(x,y))  # Not implemented

numba_mesh(J)

# -----------------------------------------------------------------------
# ----------------------------- time loop -------------------------------
# This shows how the looping works
J = 2
S = 2
D = 3
dim = tuple(np.repeat([D+1, s+1], J))
size = np.prod(dim)

S_states = array(list(product(arange(S+1), repeat=J)))
S_states = S_states[np.sum(S_states, axis=1) <= S]
x_states = array(list(product(arange(D+1), repeat=J)))

sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

for s in S_states:
    for x in x_states:
        state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
        print(x, s, state)

# Numba and classes clash, as Numba cannot work with self
# Solution, @staticmethod and pass all variables as parameters
from numpy import arange, array, prod, dot, zeros
import numpy as np
from itertools import product
from numba import njit

class Tmp():
    def __init__(self):
        self.J = 2
        self.S = 2
        self.D = 3
        self.dim = tuple(np.repeat([self.D+1, self.S+1], self.J))
        self.size = np.prod(self.dim)
        self.memory = zeros(self.dim)
        self.sizes = self.def_sizes()
        self.S_states = array(list(product(arange(self.S+1), repeat=self.J)))
        self.S_states = self.S_states[np.sum(self.S_states, axis=1) <= self.S]
        self.x_states = array(list(product(arange(self.D+1), repeat=self.J)))

        print(self.class_test_loop(self.memory, self.J, self.S, self.size,
                                    self.sizes, self.dim, self.S_states,
                                    self.x_states))

    def def_sizes(self):
        """Docstring."""
        sizes = np.zeros(len(self.dim), dtype=np.int64)
        sizes[-1] = 1
        for i in range(len(self.dim) - 2, -1, -1):
            sizes[i] = sizes[i + 1] * self.dim[i + 1]
        return sizes

    @staticmethod
    @njit
    def class_test_loop(memory, J, S, size, sizes, dim, S_states, x_states):
        """Docstring."""
        memory = memory.reshape(size)
        for s in S_states:
            for x in x_states:
                state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
                print(state)
                memory[state] = np.random.rand()
        return memory.reshape(dim)


tmp = Tmp()

# Oustide classes, the code will simply work as demostrated below.
J = 2
S = 2
D = 3
dim = (D+1, S+1)*J
size = np.prod(dim)
memory = zeros(dim)
S_states = array(list(product(arange(S+1), repeat=J)))
S_states = S_states[np.sum(S_states, axis=1) <= S]
x_states = array(list(product(arange(D+1), repeat=J)))

sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

@njit
def test_loop(memory):
    """Docstring."""
    memory = memory.reshape(size)
    for s in S_states:
        for x in x_states:
            state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
            memory[state] = np.random.rand()
    return memory

print(test_loop(memory))

# -----------------------------------------------------------------------
# ----------------------------- If Any  ---------------------------------
import numpy as np
from numba import njit
A = 4

@njit
def if_any1(x):
    """Docstring."""
    return x + A if 5<A else x

@njit
def if_any2(x):
    """Docstring."""
    return True if np.any(x==A) else False

@njit
def if_any3(x):
    """Docstring."""
    return 5+A if x[0] < A else 5

x = np.arange(5)
print(if_any1(x))  # Numba can handle one line if statements
print(if_any2(x))  # np.any works! (base any does not)
print(if_any3(x))  # This works
