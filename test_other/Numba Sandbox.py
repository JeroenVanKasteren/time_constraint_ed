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

import timeit
from numpy import arange, array, prod, dot, zeros
import numpy as np
from itertools import product
from numba import njit
import numba as nb

# ---------------------------- ARITHMATIC -----------------------------
@njit
def f_modulo():
    """modulo."""
    x = int(0)
    return 12 % 5 == 2


print(f_modulo())  # works

# ---------------------------- TUPLES CREATION -----------------------------
# from Tools_FL import create_tuple_creator, unravel_index, ravel_multi_index

# get all x_i or s_i from multi_state tuple
D = 2
s = 1
J = 2


@njit
def f1(D, s, J):
    """Create tuples."""
    dim = (D + 1, s + 1) * J  # Not possible to create tuples like this
    return dim


@njit
def f2(D, s, J):
    """Create tuples."""
    dim = np.repeat((D + 1, s + 1), J)
    return dim


# nb_tuple = create_tuple_creator(J*2)


@njit
def f3(D, s, J):
    """Create tuples."""
    dim = np.repeat((D + 1, s + 1), J)
    return nb_tuple(dim)


f1(D, s, J)  # Error
f2(D, s, J)  # Returns array, not tuple
f3(D, s, J)  # Returns tuple


@njit
def f4(D, s, J):
    """Create matrix."""
    dim = np.repeat([D + 1, s + 1], J)
    return np.zeros(dim)


@njit
def f5(D, s, J):
    """Create matrix."""
    dim = np.repeat((D + 1, s + 1), J)
    return np.zeros(dim)


f4(D, s, J)  # Error (would work in python)
f5(D, s, J)  # Error (would work in python)


# -----------------------------------------------------------------------
# ---------------------------- ELEMENTWISE --------------------------

@njit
def f5a(x, gamma, t):
    """Array creation and bitwise comparing."""
    return sum(x > gamma * t)

@njit
def f5b(x, t):
    """Array creation and bitwise comparing."""
    return x * (t - np.array([0, 0, 1, 0]))

f5a(x=array([10, 5, 4, 8]), gamma=1, t=array([2, 2, 2, 2]))
f5b(x=array([10, 5, 4, 8]), t=array([2, 2, 2, 2]))

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
    dim = np.repeat((D + 1, s + 1), J)
    print(np.array(dim)[range(0, J * 2, 2)])  # x_i
    print(np.array(dim)[arange(1, J * 2, 2)])  # s_i


@njit
def f8(D, s, J):
    """Slicing array."""
    dim = np.repeat((D + 1, s + 1), J)
    print(dim[slice(0, J)])  # x_i, need to use slice to subset tuple
    print(dim[slice(J, J * 2)])  # s_i


f7(D, s, J)  # Error (would work in python)
f8(D, s, J)  # Works


@njit
def f9(D, s, J):
    """Slicing tuple, python needs slice to subset tuple."""
    dim = nb_tuple(np.repeat((D + 1, s + 1), J))
    print(dim[slice(0, J)])  # x_i
    print(dim[slice(J, J * 2)])  # s_i


@njit
def f10(D, s, J):
    """Slicing tuple, python needs slice to subset tuple."""
    dim = nb_tuple(np.repeat((D + 1, s + 1), J))
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


A = np.arange((D + 1) ** 2 * (s + 1) ** 2).reshape([D + 1, D + 1, s + 1, s + 1])
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
    print(A[states] * A[states])


@njit
def sum_(A, states):
    """Docstring."""
    print(A[states] + A[states])


@njit
def sumproduct(A, states):
    """Docstring."""
    print(dot(A[states], (A[states])))


A = np.arange((D + 1) ** 2 * (s + 1) ** 2).reshape([D + 1, D + 1, s + 1, s + 1])
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


states = array([[2, 3], [8, 9]])
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

@njit
def global_test_3(x):
    """Docstring."""
    V_t = 5 * x
    V_t[0] = -1
    return V_t

x = np.arange(10)
print(global_test_3(x), x)

# -----------------------------------------------------------------------
# -------------------------- Looping in Numba ---------------------------

X = np.arange(1e7)

@nb.njit
def loop_X(X):
    res = 0
    for x in X:
        res += x / 1000 + 42


@nb.njit
def loop_len(X):
    res = 0
    for i in arange(len(X)):
        res += X[i] / 1000 + 42


@nb.njit
def loop(X):
    res = 0
    n = len(X)
    for i in arange(n):
        res += X[i] / 1000 + 42


@nb.njit(parallel=True)
def loop_p(X):
    res = 0
    n = len(X)
    for i in nb.prange(n):
        res += X[i] / 1000 + 42


N = int(1e2)
start_time = timeit.default_timer()
for i in range(N):
    loop_X(X)
print((timeit.default_timer() - start_time) / N)

start_time = timeit.default_timer()
for i in range(N):
    loop_len(X)
print((timeit.default_timer() - start_time) / N)

start_time = timeit.default_timer()
for i in range(N):
    loop_len(X)
print((timeit.default_timer() - start_time) / N)

start_time = timeit.default_timer()
for i in range(N):
    loop_p(X)
print((timeit.default_timer() - start_time) / N)

#Take away, numba optimizations do not work for small functions

J = 2
s = 2
D = 3
dim = tuple(np.repeat([D + 1, s + 1], J))
size = np.prod(dim)


@njit
def states_f(s_states, x_states, J, order):
    for s_state in s_states:
        if (np.sum(s_state) > s):
            continue
        for x_state in x_states:
            lst = [x_state, s_state]
            flattend = array([x for sublst in lst for x in sublst])
            tmp = flattend[order]
            print(tmp)


order = np.ravel([arange(0, J * 2, 2), arange(1, J * 2, 2)])
s_states = array(list(product(arange(s + 1), repeat=J)))
x_states = array(list(product(arange(D + 1), repeat=J)))
states_f(s_states, x_states, J, order)

# The following statements do not work in Numba
# As creating an array from an array(s) is not possible
# np.concatenate(tmp)
# index = np.ravel([arange(0, J*2, 2), arange(1, J*2, 2)])
# array(tmp).reshape(-1)

s_states = array(list(product(arange(s + 1), repeat=J)))
s_states = s_states[np.sum(s_states, axis=1) <= s]
x_states = array(list(product(arange(D + 1), repeat=J)))

# -----------------------------------------------------------------------
# ----------------------------- RAVELLING -------------------------------

J = 2
s = 2
D = 3
dim = np.repeat([D + 1, s + 1], J)
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
dim = tuple(np.repeat([D + 1, s + 1], J))
size = np.prod(dim)

S_states = array(list(product(arange(S + 1), repeat=J)))
S_states = S_states[np.sum(S_states, axis=1) <= S]
x_states = array(list(product(arange(D + 1), repeat=J)))

sizes = np.zeros(len(dim), dtype=np.int64)
sizes[-1] = 1
for i in range(len(dim) - 2, -1, -1):
    sizes[i] = sizes[i + 1] * dim[i + 1]

for s in S_states:
    for x in x_states:
        state = np.sum(x * sizes[0:J] + s * sizes[J:J * 2])
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
        self.dim = tuple(np.repeat([self.D + 1, self.S + 1], self.J))
        self.size = np.prod(self.dim)
        self.memory = zeros(self.dim)
        self.sizes = self.def_sizes()
        self.S_states = array(list(product(arange(self.S + 1), repeat=self.J)))
        self.S_states = self.S_states[np.sum(self.S_states, axis=1) <= self.S]
        self.x_states = array(list(product(arange(self.D + 1), repeat=self.J)))

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
                state = np.sum(x * sizes[0:J] + s * sizes[J:J * 2])
                print(state)
                memory[state] = np.random.rand()
        return memory.reshape(dim)


tmp = Tmp()

# Oustide classes, the code will simply work as demostrated below.
J = 2
S = 2
D = 3
dim = (D + 1, S + 1) * J
size = np.prod(dim)
memory = zeros(dim)
S_states = array(list(product(arange(S + 1), repeat=J)))
S_states = S_states[np.sum(S_states, axis=1) <= S]
x_states = array(list(product(arange(D + 1), repeat=J)))

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
            state = np.sum(x * sizes[0:J] + s * sizes[J:J * 2])
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
    return x + A if 5 < A else x


@njit
def if_any2(x):
    """Docstring."""
    return True if np.any(x == A) else False


@njit
def if_any3(x):
    """Docstring."""
    return 5 + A if x[0] < A else 5

@njit
def sum_bools(x):
    """Docstring."""
    return sum(np.arange(5) < 3), sum(np.arange(5) < np.repeat(3, 5))

x = np.arange(5)
print(if_any1(x))  # Numba can handle one line if statements
print(if_any2(x))  # np.any works! (base any does not)
print(if_any3(x))  # This works
print(sum_bools(x))  # This works


# -----------------------------------------------------------------------
# ----------------------------- Dictionairy -----------------------------

import numpy as np
import numba as nb
from numba.typed import Dict
from numba import types as tp

c = 2
vector_f = np.array([1.1, 1.2], np.float32)
vector_i = np.array([1, 1], int)

d = {'c': c, 'vector_f': vector_f, 'vector_i':vector_i}
d_ints = {'1':  1, '2': 2}
d_i = Dict.empty(key_type=tp.unicode_type, value_type=tp.i8)
d_f = Dict.empty(key_type=tp.unicode_type, value_type=tp.f8)
d_i['1'] = 1
d_i['2'] = 2
d_f['a'] = 1.1
d_f['b'] = 2.2

@nb.njit
def dict_python(d):
    return d['vector_f']

@nb.njit
def dict_int(d):
    return d['1']

print(dict_python(d))  # works in python mode, not in no-python mode
print(dict_int(d_ints))  # works in python mode, not in no-python mode

dict_ty = tp.DictType(tp.unicode_type, tp.i8)

@nb.njit (tp.f8[:](tp.f8[:], dict_ty))
def dict_numba(x, d):
    x[3] = d['1']
    x[4] = d['2']
    return x

x = np.arange(5, dtype=float)
print(dict_numba(x, d_i))

# -----------------------------------------------------------------------
# ----------------------------- Importance of signitures ----------------

import timeit
import numba as nb
import numpy as np

@nb.njit  # (nb.int64(nb.int64))
def cube_formula(x):
    return x**3 + 3*x**2 + 3

@nb.njit # (nb.int64[:](nb.int64[:]))
def perform_operation_jitted(x):
    out = np.empty_like(x)
    for i, elem in enumerate(x):
        res = cube_formula(elem)
        out[i] = res
    return out

print(np.mean(timeit.repeat("perform_operation_jitted(np.arange(1e6, dtype=np.int64))",
                    "from __main__ import perform_operation_jitted," 
                    "cube_formula; import numpy as np",
                    repeat=10, number=1)))

# without numba: 1.76
# with numba: 0.027
# with numba & handles: 0.0092

# -----------------------------------------------------------------------
# ------------------------------- Array creation ------------------------

import timeit
import numpy as np
import numba as nb
from numba import types as tp

DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float vector

@nb.njit((DICT_TYPE_F, tp.i8))
def copy_numba(d_f, J):
    # x = np.empty(J, nb.f8)
    for _ in range(1e5):
        x = d_f['x'].copy()  # Without copy does not copy
        # x[0] = 2
        # print(x, d_f)

@nb.njit((DICT_TYPE_F, tp.i8))
def copy_numba(d_f, J):
    x = np.empty(J, nb.f8)
    for _ in range(1e5):
        for i in range(J):
            x[i] = d_f['x'][i]  # Copies
            # x[0] = 2
            # print(x, d_f)

# 0.02624, however, also parallelizing the inner loop makes it slower (0.0312)
@nb.njit((DICT_TYPE_F, tp.i8), parallel=True, error_model='numpy')
def copy_numba(d_f, J):
    x = np.empty(J, nb.f8)
    for _ in nb.prange(1e5):
        # for i in np.arange(J):
        for i in nb.prange(J):
            x[i] = d_f['x'][i]  # Copies

# Approx as fast (without parallel 0.02915, with parallel 0.02614)
# error model numpy has only a marginal effect of 1% faster
@nb.njit((DICT_TYPE_F, tp.i8), parallel=True, error_model='numpy')
def copy_numba(d_f, J):
    for _ in np.arange(1e5):
    # for _ in nb.prange(1e5):
        x = d_f['x'].copy()  # Copies

# Subsetting test, same time
@nb.njit((DICT_TYPE_F, tp.i8), parallel=True, error_model='numpy')
def copy_numba(d_f, J):
    y = d_f['x'][1:2]
    for _ in nb.prange(1e5):
        x = y.copy()  # Copies

# Subsetting test, same time
@nb.njit((DICT_TYPE_F, tp.i8), parallel=True, error_model='numpy')
def copy_numba(d_f, J):
    for _ in nb.prange(1e5):
        x = d_f['x'][1:2].copy()

d_f = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.f8[:])
d_f['x'] = np.array([2.4, 2.5, 2.6])

copy_numba(d_f, len(d_f['x']))

print(np.mean(timeit.repeat("copy_numba(d_f, len(d_f['x']))",
                    "from __main__ import copy_numba," 
                    "DICT_TYPE_F, d_f; import numpy as np; import numba as nb",
                    repeat=20, number=5))/5)

# -----------------------------------------------------------------------
# ------------------------------- sum sizes ------------------------

import timeit
import numpy as np
import numba as nb
from numba import types as tp
from itertools import product

DICT_TYPE_I = tp.DictType(tp.unicode_type, tp.i4[:])  # int vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # float vector

D = 100
J = 2
S = 10
dim_i = tuple(np.append(J + 1, np.repeat([D + 1, S + 1], J)))
d_i = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.i4[:])
d_i2 = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.i4[:, :])
s_states = np.array(list(product(np.arange(S + 1), repeat=J)))
d_i2['s'] = s_states[np.sum(s_states, axis=1) < S]
d_i2['x'] = np.array(list(product(np.arange(D + 1), repeat=J)))
sizes_i = np.zeros(len(dim_i), int)
sizes_i[-1] = 1
for i in range(len(dim_i) - 2, -1, -1):
    sizes_i[i] = sizes_i[i + 1] * dim_i[i + 1]
d_i['sizes_i'] = sizes_i


@nb.njit((tp.i8, DICT_TYPE_I, DICT_TYPE_I2)) #, parallel=True)
def w_numba(J, d_i, d_i2):
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    for s_i in nb.prange(len(d_i2['s'])):
        for x_i in nb.prange(len(d_i2['x'])):
            for i in nb.prange(J):
                x = d_i2['x'][x_i].copy()
                s = d_i2['s'][s_i].copy()
                state = i * d_i['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
# 0.974s without parallel, 0.7648s with parallel, without numba 57.5s
# not copying: 0.592s
# no function handles 1.129

@nb.njit((tp.i8, DICT_TYPE_I, DICT_TYPE_I2), parallel=True)
def w_numba(J, d_i, d_i2):
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    for s_i in nb.prange(len(d_i2['s'])):
        for x_i in nb.prange(len(d_i2['x'])):
            for i in nb.prange(J):
                x = d_i2['x'][x_i].copy()
                s = d_i2['s'][s_i].copy()
                state = i * d_i['sizes_i'][0]
                for j in np.arange(J):
                     state += x[j] * sizes_x[j] + s[j] * sizes_s[j]
# 0.809 with parallel, not faster

w_numba(J, d_i, d_i2)

print(np.mean(timeit.repeat("w_numba(J, d_i, d_i2)",
                    "from __main__ import w_numba," 
                    "DICT_TYPE_I, DICT_TYPE_I2, J, d_i, d_i2;"
                    "import numpy as np; import numba as nb",
                    repeat=5, number=3))/3)


# ------------------------- SIMULATION ---------------------------

'''
Key things to consider:
The Numba type of a typed.List is a types.ListType.
One cannot call typeof from inside a jitted region, 
(Workaround, mytype = typeof(mytype_instance) globally and refer to mytype)
typed.List.empty_list and typed.Dict.empty take Numba types as arguments.
'''

import numba as nb
import heapq as hq

entry_type = nb.typeof((0.0, 0, 'event'))

@nb.njit
def heapsort(iterable):
    time = 0
    heap = nb.typed.List.empty_list(entry_type)
    for i in range(len(iterable)):
        hq.heappush(heap, (iterable[i], 0, 'arrival'))
        time += iterable[i]
    return heap, time


@nb.njit
def heapsort2(iterable):
    time = 0
    heap = nb.typed.List.empty_list(entry_type)
    # heap = []
    for i in range(len(iterable)):
        hq.heappush(heap, (iterable[i][0], 0, 'arrival'))
        time += iterable[i][1]
    return heap, time


x = nb.typed.List([1.232, 3.21, 5.21, 7.54, 9.765, 2.35, 4.85, 6.00, 8.1, 0.23])
print(heapsort(x))


y = [[1.232, 3.21, 5.21], [7.54, 9.765, 2.35], [4.85, 6.00, 8.1]]
iterable = nb.typed.List()
for i in range(len(y)):
    iterable.append(nb.typed.List(y[i]))
print(heapsort2(iterable))

# ------------------------- Timing ---------------------------
import numba as nb
from time import perf_counter as clock

start_time = clock()


@nb.njit
def my_numba_function():
    with nb.objmode():
        print('time: {}'.format(start_time - clock()))

my_numba_function()
