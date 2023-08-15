"""
Take home.

For variable multidimensional matrices, tuples are used (as explains below)
https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
However, this is not possible in numba

# Supported functions
https://numba.pydata.org/numba-doc/dev/reference/pysupported.html

# (un)ravel
Explanation:
https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
Note that np.flatten() returns a copy and np.ravel() returns a view.
Ravel is faster, but keeps the link to the original object!
(reshape is similar to ravel, but in some cases returns a copy)

unravel_index(index=i, shape=dim) given a matrix of shape dim,
gives multi_index of index (index = place in the matrix, single number).

ravel_multi_index
"""

import numpy as np
from numba import njit

D = 2
s = 1
J = 2
dim = np.repeat((D+1, s+1), J)
A = np.arange((D+1)**2 * (s+1)**2).reshape([D+1, D+1, s+1, s+1])

# All three below are correct
print(A.size)
print(np.prod(dim))
print(((D+1)*(s+1))**J)


@njit
def unravel_index(index, shape):
    """
    Return multi_index, given index of element and shape of matrix.

    How does unravel_index work?
    Returns the multi_index given the shape of a matrix and the index/place of
    an element. The index/place is viewed as indexing in row-major (C-style).

    Sizes is per dimension the size of every element/submatrix. The size is the
    number of elements in every element/submatrix. In other words, if you would
    select an element/submatrix along a dimension, what would the size be of
    this element/submatrix.

    Next, the result is a multi_index that represents per dimension which
    element/index contains the requested place/element/index.
    This is calculated by dividing the place/element/index/remainder by the
    size of the dimension and taking the integer part (the // operator)
    Next, the remainder is calculated with modulo, to help to determine the
    place/element/index of the next dimension.
    """
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return result


@njit
def ravel_multi_index(multi_index, dims):
    """
    Return place of multi_index given shape of matrix.

    How does ravel_multi_index work?
    Returns the index/place of an element multi_index given the shape of a
    matrix. The index/place is viewed as indexing in row-major (C-style).

    Sizes is per dimension the size of every element/submatrix. The size is
    the number of elements in every element/submatrix. In other words, if you
    would select an element/submatrix along a dimension, what would the size
    be of this element/submatrix.

    Next, the result is determined by summing multi_index * sizes.
    """
    sizes = np.zeros(len(dims), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(dims) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dims[i + 1]
    return np.sum(sizes * multi_index)


@njit
def size_per_dim(dims):
    """
    Return array with size per dimension.

    Sizes is per dimension the size of every element/submatrix. The size is
    the number of elements in every element/submatrix. In other words, if you
    would select an element/submatrix along a dimension, what would the size
    be of this element/submatrix.
    """
    sizes = np.zeros(len(dims), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(dims) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dims[i + 1]
    return sizes


@njit
def numba_jit(multi_index, sizes):
    """Numba test code."""
    return np.sum(sizes * multi_index, axis=1)


# (un)ravel
place = 1  # index = place
multi_index = np.unravel_index(place, shape=dim)
value = np.ravel_multi_index(multi_index, dims=dim)
print(multi_index, unravel_index(place, dim),
      value, ravel_multi_index(np.array(multi_index), dim))

multi_index = np.array([[1, 1, 1, 1], [1, 0, 0, 1]])
print(A[tuple(multi_index[0])])
print(A[tuple(multi_index[1])])
dims = dim
sizes = size_per_dim(dim)
print(numba_jit(multi_index, sizes))
