"""
Description.

# -s <= x < 0,
g + tau*V(x) = lmbda * (r + V(x+1)) + (x + S)*mu*V(max(x-1,0)) + \
                (tau - lmbda - (x+S)*mu)*V(x)
note V[max(x+S-1,0)] maps V(-s-1) -> V(-s)=0, therefore term is ignored
Moreover, when x=-s then (x+S)*mu=0
"""

import numpy as np
from numpy import array, arange, maximum, zeros, around, exp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

from OtherTests.init import Env

np.set_printoptions(precision=4, linewidth=150, suppress=True)
env = Env(J=1, S=1, mu=array([2]), lmbda=array([1]), t=array([2]),
          r=array([1]), c=array([1]), P=0,
          gamma=2, D=6, e=1e-5, trace=False)

def get_pi_0(env, s, rho):
    """Calculate pi(0)."""
    pi_0 = s*exp(s*rho) / (s*rho)**s * gamma_fun(s)*reg_up_inc_gamma(s,s*rho)
    pi_0 += (env.gamma + rho * env.lmbda)/env.gamma * (1 / (1 - rho))
    return 1 / pi_0

def get_tail_prob(env, s, rho, pi_0):
    """P(W>t)."""
    tail_prob = pi_0/(1-rho) * \
        (env.lmbda+env.gamma) / (env.gamma + env.lmbda*pi_0) * \
        (1-(s*env.mu-env.lmbda) / (s*env.mu+env.gamma))**(env.gamma*env.t)
    return tail_prob

def V_f(env, g):
    """Calculate V for a single queue."""
    S=env.S; lmbda=env.lmbda; mu=env.mu; gamma=env.gamma; rho=env.rho; r=env.r; a=env.a
    V = zeros(S + env.D + 1)

    x = arange(-S+1, +1)  # V(x) for x<=0, with V(-s)=0
    V_x_le_0 = lambda y: (1-(y/a)**(x+S))/(1-y/a)*exp(a-y)
    V[x+S] = (g-lmbda*r)/lmbda * quad_vec(V_x_le_0, a, np.inf)[0]

    frac = (S * mu + gamma) / (lmbda + gamma)
    trm = exp(a) / a ** (S - 1) * gamma_fun(S) * reg_up_inc_gamma(S, a)
    x = arange(1, env.D + 1)  # V(x) for x>0
    # x = arange(0, env.D + 1)  # V(x) for x>=0
    V[x+S] = V[S] + (S*mu*r - g) / (gamma*S*mu*(1 - rho)**2) * \
        (lmbda + gamma - lmbda*x*(rho-1) - (lmbda+gamma)*frac**x) + \
            1/(gamma*(rho-1)) * (g - S*mu*r - gamma/lmbda *
                                 (g + (g-lmbda*r)/rho * trm)) * (-rho + frac**(x-1))
    # 1 if x=0
    # V[S] += 1 / (gamma * (rho - 1)) * (g - S * mu * env.r - gamma / lmbda *
    #                                    (g + (g - lmbda * r) / rho * trm)) * (gamma * (rho - 1) / (S * mu + gamma))

    # -1_{x > gamma*t}[...]
    x = arange(gamma*env.t+1, env.D+1).astype(int)
    V[x+S] -= env.c / (gamma * (1 - rho)**2) * \
        (lmbda + gamma - lmbda*(x - gamma*env.t - 1) * (rho - 1) -
         (lmbda + gamma) * frac**(x-gamma*env.t-1))
    return V


pi_0 = get_pi_0(env, env.s_star, env.rho)
tail_prob = get_tail_prob(env, env.S, env.rho, pi_0)
g = (env.r - env.c*tail_prob) * (env.lmbda + pi_0*env.lmbda**2/env.gamma)
P_xy = env.P_xy[0]
V = V_f(env, g)

S=env.S; D=env.D; lmbda=env.lmbda; mu=env.mu; gamma=env.gamma; rho=env.rho

x = arange(-S,D+1)
LHS = g + env.tau*V[x+S]
RHS = np.zeros(S+D+1)
x = arange(-S,0)  # -s <= x < 0
RHS[x+S] = lmbda*(env.r+V[x+S+1]) + (x + S)*mu*V[maximum(x+S-1,0)] + \
    (env.tau - lmbda - (x+S)*mu)*V[x+S]
x = 0  # x = 0
RHS[x+S] = lmbda*V[x+S+1] + (x + S)*mu*V[maximum(x+S-1,0)] + \
    (env.tau - lmbda - (x+S)*mu)*V[x+S]
x = arange(1,D)  # x>=1
RHS[x+S] = gamma*V[x+S+1] + \
    S*mu*(env.r + np.sum(P_xy[1:D,:D]*V[S:D+S],1)) + \
        (env.tau - gamma - S*mu)*V[x+S]
x = arange(env.t*gamma+1,D).astype(int)  # x>t*gamma
RHS[x+S] -= S*mu * env.c

print("pi_0: ", pi_0)
print("tail_prob: ", tail_prob)
print("g: ", g)
# print("g*tau: ", g*env.tau)
print("V", V)
dec = 5
x = arange(-env.S, env.D)
print("gamma*t: ", env.gamma*env.t)
print("LHS==RHS? ", (around(LHS[x+S], dec) == around(RHS[x+S], dec)).all())
print("x, LHS, RHS, V: \n", np.c_[arange(-env.S, env.D+1), LHS, RHS, V])

print(env.P_xy)
# V_single_queue = V
# V_single_queue[4:] - V[:,4]
