"""
Created on Mon May  4 10:25:06 2020.

Create tuple function for numba by

from Tools_FL import create_tuple_creator
nb_tuple = create_tuple_creator(length)
tuple = nb_tuple(array)
"""

from numpy import zeros, round, arange
from numpy import np
from numpy.random import rand
from timeit import default_timer
from sys import getsizeof, exit
from numba import njit

def feasibility(env):
    """Feasability Check matrix size and rho<1."""
    memory = zeros(env.dim)
    if env.trace:
        print("GB: ", round(getsizeof(memory)/10**9, 4))
    start = default_timer()
    test_loop(memory, env.dim)
    time = default_timer() - start
    if(time - start > 60):  # in seconds
        print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))
        exit("Looping matrix takes more than 60 seconds.")
    if env.trace:
        print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))
    rho = sum(env.lambda_) / (sum(env.mu) * env.s)
    if env.trace:
        print("J =", env.J, ", D =", env.D, ", s =", env.s, '\n',
              "lambda:", round(env.lambda_, 4), '\n',
              "mu:", round(env.mu, 4), '\n',
              "a:", round(env.lambda_/env.mu, 4), '\n',
              "rho:", round(rho, 4), '\n',
              "Target:", round(env.t, 4))
    assert rho < 1, "rho < 1 does not hold"

@njit
def test_loop(memory, dim):
    """Docstring."""
    size = np.prod(dim)
    memory = memory.reshape(size)
    for d in dim:
        for x in arange(d):
            memory[0] = rand(4)
