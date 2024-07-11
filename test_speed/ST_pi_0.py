"""
pi_0.

Take home,

Test 2.i
Creating a matrix can take more time
than doing an action on the matrix

Test 1, Integer and for loop
time 0.0001 test [0.096 0.155 0.166]

Test 2, Integer and partly vectorized
time 0.0001 test [0.096 0.155 0.166]

Test 3, Integer and vectorized
time 0.0 test [0.096 0.155 0.166]

Test 4, Analytic Continuation
time 0.0 test [0.096 0.155 0.166]

Test 2.1 factor all
10 µs ± 61.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Test 2.2 factor one
8.13 µs ± 62.6 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Test 2.3 factor with matrix already created
1.81 µs ± 8.78 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

Calculate pi(0)
Created to work with one or multiple classes
"""

from numpy import array, zeros, round, tile, maximum, arange, reshape, sum
from numpy import set_printoptions, exp
from numpy.random import binomial

from scipy.special import factorial as fac
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

from timeit import default_timer

set_printoptions(precision=3)
N = 1000

# get_pi_0(self, _s, rho, **kwargs)
J = 3
lambda_ = array([1, 1.5, 0.5])
mu = array([0.2, 1, 0.5])
s = 10
rho = sum(lambda_/mu)/s  # Total system load < 1
assert rho < 1, "rho < 1 does not hold"
lambda_/mu  # Used to estimate s_star
s_star = array([6, 2, 2])
gamma = 20
D = 50

Js = arange(J)  # Classes
# Js = Js[0]

print("J=", J, ", D=", D, ", s=", s, '\n',
      "lambda:", round(lambda_, 4), '\n',
      "mu:", round(mu, 4), '\n')

# Test 1, Integer and for loop
start = default_timer()
for n in arange(N):
    pi_0 = zeros(J)
    for i in Js:
        _lambda = lambda_[i]
        _mu = mu[i]
        _s = s_star[i]
        _rho = _lambda / (_s * _mu)
        pi_0[i] = 0
        for k in arange(_s-1+1):  # Sum
            pi_0[i] += (_s*_rho)**k / fac(k)
        pi_0[i] += (_s*_rho)**_s / fac(_s) * \
            (gamma + _rho * _lambda)/gamma * (1 / (1 - _rho))
        pi_0[i] = 1 / pi_0[i]
        pi_0[i] *= (_s * _rho)**_s / fac(_s)
print('Test 1, Integer and for loop')
print('time', round((default_timer()-start)/N, 4),
      'test', round(pi_0, 4), '\n')

# Test 2, Integer and partly vectorized
start = default_timer()
for n in arange(N):
    _s = s_star
    _rho = lambda_ / (_s*mu)
    pi_0 = zeros(J)
    for i in Js:  # For every class
        k = arange(_s[i]-1+1)
        pi_0[i] += sum((_s[i]*_rho[i])**k / fac(k))
    pi_0 += (_s*_rho)**_s / fac(_s) * \
        (gamma + _rho * lambda_)/gamma * (1 / (1 - _rho))
    pi_0 = 1 / pi_0
    pi_0 *= (_s * _rho)**_s / fac(_s)
print('Test 2, Integer and partly vectorized')
print('time', round((default_timer()-start)/N, 4),
      'test', round(pi_0, 4).reshape(J), '\n')

# Test 3, Integer and vectorized
start = default_timer()
for n in arange(N):
    _lambda = lambda_.reshape(J, 1)
    _mu = mu.reshape(J, 1)
    _s = s_star.reshape(J, 1)
    _rho = _lambda / (_s * _mu)
    A = tile(arange(max(_s)-1+1), (J, 1))
    mask = A < _s
    pi_0 = sum((_s*_rho)**A / fac(A) * mask, axis=1).reshape(J, 1)
    pi_0 += (_s*_rho)**_s / fac(_s) * \
        (gamma + _rho * _lambda)/gamma * (1 / (1 - _rho))
    pi_0 = 1 / pi_0
    pi_0 *= (_s * _rho)**_s / fac(_s)
print('Test 3, Integer and vectorized')
print('time', round((default_timer()-start)/N, 4),
      'test', round(pi_0, 4).reshape(J), '\n')

# Test 4, Analytic Continuation
start = default_timer()
for n in arange(N):
    _rho = lambda_ / (s_star * mu)
    a = lambda_ / mu
    pi_0 = s_star * exp(a) / a**s_star * \
        gamma_fun(s_star) * reg_up_inc_gamma(s_star, a)
    pi_0 += (gamma + _rho * lambda_)/gamma * (1 / (1 - _rho))
    pi_0 = 1 / pi_0
print('Test 4, Analytic Continuation')
print('time', round((default_timer()-start)/N, 4),
      'test', round(pi_0, 4), '\n')

# Test 2.i factor matrix or vector
J = 10
s_star = binomial(10, 0.5, J)

print('Test 2.1 factor all')
# %timeit fac(tile(arange(max(s_star)-1+1), (J, 1)))

print('\nTest 2.2 factor one')
# %timeit tile(fac(arange(max(s_star)-1+1)), (J, 1))

print('\nTest 2.3 factor with matrix already created')
A = tile(arange(max(s_star)-1+1), (J, 1))
# %timeit fac(A)
