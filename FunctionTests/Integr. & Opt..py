"""
Take home.

Calculations with arrays and returning arrays
Using double sum in V(x) as example
Showing that quad works for numbers and quad_vec for arrays

Integration versus double sum
Double sum is 30x times faster.
"""

from numpy import arange, exp, inf, round
import numpy as np
from scipy.special import factorial as fac
from scipy.integrate import quad, quad_vec
from scipy import optimize
import matplotlib.pyplot as plt
import timeit

S = 4
x = np.arange(-S, 0+1)
a = 3
N = 500


def f1(y):
    """Docstring."""
    return x * exp(-y) * S*a


def f2(y):
    """Docstring."""
    return 3 * exp(-y) * S*a


def V_x_le_0_(y):
    """Number."""
    return (1 - (y/a)**(3+S)) / (1 - y/a) * exp(a-y)


def V_x_le_0(y):
    """Array."""
    return (1 - (y/a)**(x+S)) / (1 - y/a) * exp(a-y)


print(V_x_le_0(0))
print(V_x_le_0_(0))

quad(f2, 0, inf)[0]
# quad(f1, 0, inf)[0]  # ERROR
quad_vec(f1, 0, inf)[0]

quad(V_x_le_0_, a, inf)[0]
quad_vec(V_x_le_0, a, inf)[0]

V_x_le_0 = lambda y: (1-(y/a)**(x+S))/(1-y/a)*np.exp(a-y)
quad_vec(V_x_le_0, a, inf)[0]

# Test 1, Double sum
start = timeit.default_timer()
for n in arange(N):
    V = np.zeros(len(x))
    A = np.delete(np.indices((S+1, S+1)), 0, 1)  # Indices Matrix i,j
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = fac(A[0] - 1) / fac(A[0] - A[1] - 1) * (1/a)**A[1]
    # Double sum
    for k in arange(1, S+1):
        V[k] = V[k-1] + sum(tmp[k-1, 0:k-1+1])
print('Test 1, one class')
print('time', round((timeit.default_timer()-start)/N, 4),
      'test', round(V, 4), '\n')


# Test 2, integration
start = timeit.default_timer()
for n in arange(N):
    V = quad_vec(V_x_le_0, a, np.inf)[0]
print('Test 1, one class')
print('time', round((timeit.default_timer()-start)/N, 4),
      'test', round(V, 4), '\n')

# ---------------------------- Optimization (and plotting) ----------


def f(x):
    """Docstring."""
    return -np.exp(-(x - 0.7)**2)


epsilon = 1e-6
x = np.arange(0, 2*(np.pi), 0.1)
plt.plot(x, f(x))
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('My graph!')
plt.show()

result = optimize.minimize_scalar(fun=f, tol=epsilon)
result.success  # check if solver was successful
result.x


def f2(x):
    """Rosenbrock function."""
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2


x0 = [1.3, 0.7]
result = optimize.minimize(f2, x0, tol=1e-6)
print(result)

x0 = [1.3, 0.7]
bounds = [[2, 10], [0, 100]]
result = optimize.minimize(f2, x0, tol=1e-6, bounds=bounds)
print(result)