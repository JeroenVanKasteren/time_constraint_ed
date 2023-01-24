"""
Check Analytical continuation by comparing with discrete version.
Code only works for J=1
"""

import os
import numpy as np
from numpy import array, arange, exp
from scipy.special import (factorial as fac, gamma as gamma_fun,
                           gammaincc as reg_up_inc_gamma)
from scipy.integrate import quad_vec
import matplotlib.pyplot as plt

PATH = (r"D:\Programs\Surfdrive\Surfdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")
# PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")
os.chdir(PATH+"\Other")
from init import Env

env = Env(J=1, S=3, mu=array([1]), lmbda=array([2]), t=array([0.5]),
          gamma=1, D=3, P=1e2, e=1e-5,
          max_iter=10, trace=True, print_modulo=5, time_check=True)

s_ = np.round(np.arange(1.5,6.01,0.01),4)
rho = env.lmbda / (s_*env.mu)
rho[rho == 1] = 1 - env.e  # Avoid division by 0
# Get integers, remove rho=1 (otherwise division by 0)
mask = (np.mod(s_,1)==0) & (env.lmbda / (s_ * env.mu) != 1)
s_int = s_[mask]

# ------------------- Single sum within V(x), x>=0 ------------
# Integer
V = np.zeros(len(s_int))
for x, s in enumerate(s_int):
    for i in arange(s-1+1):
        V[x] += fac(s-1) / fac(s-i-1) * (env.a)**-i
V_int = V.copy()

# Analytic Continuation
V = exp(env.a)/env.a**(s_-1) * gamma_fun(s_) * reg_up_inc_gamma(s_, env.a)

plt.plot(s_, V, label='Sum of V')
plt.scatter(s_int, V_int, label='Sum of V_int')
plt.xlabel('s')
plt.ylabel('y')
plt.title('Analytical Continuation V x=>0')
plt.legend()
plt.show()

print("V == V_int: ", np.allclose(V[mask], V_int, rtol=1e-10, atol=0))

# ------------------- pi(0), single sum ------------
# Integer
pi_0 = s_int.copy()
for i, s in enumerate(s_int):
    rho_ = env.lmbda / (s * env.mu)
    pi_0[i] = 0
    for k in arange(s-1+1):  # Sum i=0 to s-1, +1 for up and until
        pi_0[i] += (s*rho_)**k / fac(k)
    pi_0[i] += (s*rho_)**s / fac(s) * \
        (env.gamma + rho_ * env.lmbda)/env.gamma * (1 / (1 - rho_))
    pi_0[i] = 1 / pi_0[i] * (s*rho_)**s / fac(s)
pi_0_int = pi_0.copy()

# Analytic Continuation
pi_0 = s_*exp(s_*rho) / (s_*rho)**s_ * gamma_fun(s_) * \
    reg_up_inc_gamma(s_, env.a)
pi_0 += (env.gamma + rho * env.lmbda)/env.gamma * (1 / (1 - rho))
pi_0 = 1 / pi_0

def get_tail_prob(env, s_, rho, pi_0):
    """P(W>t)."""
    tail_prob = pi_0/(1-rho) * (env.lmbda+env.gamma) / \
        (env.gamma + env.lmbda*pi_0) * \
        (1 - (s_*env.mu-env.lmbda)/
         (s_*env.mu+env.gamma))**(env.gamma*env.t)
    return tail_prob

tail_prob = get_tail_prob(env, s_, rho, pi_0)

plt.plot(s_, tail_prob, label='tail_prob')
plt.plot(s_, rho, label='rho')
plt.axhline(y=1,color='r')
plt.plot(s_, pi_0, label='pi_0')
plt.scatter(s_int, pi_0_int, label='pi_0_int')
plt.xlabel('s')
plt.ylabel('y - axis')
plt.title('Analytical Continuation pi(0)')
plt.legend()
plt.show()

print("pi(0) == pi(0)_int: ",
      np.allclose(pi_0[mask], pi_0_int, rtol=1e-10, atol=0))

# ------------------- V(x), x<=0 ------------
# Integer
x = -2

V = np.zeros(len(s_int))
for k, s in enumerate(s_int):
    for i in arange(1, x+s+1):  # i=0 -> x+s
        for j in arange(i-1+1):
            V[k] += fac(i-1) / fac(i-j-1) * (env.a)**-j
V_int = V.copy()

# Analytic Continuation
integral = lambda y: (1-(y/env.a)**(x+s_)) / (1-y/env.a) * np.exp(env.a-y)
V = quad_vec(integral, env.a, np.inf)[0]

plt.plot(s_, V, label='Sum of V')
plt.scatter(s_int, V_int, label='Sum of V_int')
plt.xlabel('s')
plt.ylabel('y')
plt.title('Analytical Continuation V x=<0 (x=-2)')
plt.legend()
plt.show()

print("V == V_int: ", np.allclose(V[mask], V_int, rtol=1e-10, atol=0))
