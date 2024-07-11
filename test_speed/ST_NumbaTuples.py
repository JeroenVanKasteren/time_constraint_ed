"""
Tuple creation.

Take home,
code from github is fast!

Both are done within ~1 micro second!
# https://github.com/numba/numba/issues/2771#ref-issue-434901811
"""

from numba import njit
from numpy import arange


def create_tuple_creator(n):
    """Docstring."""
    assert n > 0
    f = njit(lambda i, x: x[i])

    @njit
    def creator(args):
        return (f(0, *args),)
    for i in range(1, n):
        @njit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)
    return njit(lambda *args: creator(args))


length = 10**2
state = arange(length)*2 + 1
nb_tuple = create_tuple_creator(n=length)

print('Test 1 tuple()')
# %timeit tuple(state)

print('Test 2 nb_tuple()')
# %timeit nb_tuple(state)

# Prove it works in numba
length = 5
state = arange(length)*2 + 1
nb_tuple = create_tuple_creator(length)


@njit
def test(state):
    """Docstring."""
    return nb_tuple(state)


print(test(state))
