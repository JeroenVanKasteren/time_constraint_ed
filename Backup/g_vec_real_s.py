# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:49:09 2020.

@author: Jeroen.
"""

import numpy as np
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

J = 3
lambda_V = np.array([1, 1.5, 0.5])
mu_V = np.array([0.5, 1, 0.5])
s_V = 7
print(sum(lambda_V/mu_V)/s_V)  # Total system load < 1
print(lambda_V/mu_V)  # Used to estimate s_star
s_star = np.array([3, 2, 2])
t_V = np.array([4]*J)
gamma = 1
tau_V = np.maximum(lambda_V, gamma) + s_star*mu_V
D = 10
weight_V = np.array([1, 3, 1])


def get_pi_0(_s, rho, **kwargs):
    """Calculate pi(0)."""
    Js = kwargs.get('i', range(J))  # Classes
    lambda_ = lambda_V[Js]  #; gamma = self.gamma
    a = _s*rho
    pi_0 = _s * np.exp(a) / a**_s * \
        gamma_fun(_s) * reg_up_inc_gamma(_s, a)
    pi_0 += (gamma + rho * lambda_)/gamma * (1 / (1 - rho))
    return 1 / pi_0


def get_tail_prob(_s, rho, pi_0, **kwargs):
    """P(W>t)."""
    Js = kwargs.get('i', range(J))
    lambda_ = lambda_V[Js]; mu = mu_V[Js]; t = t_V[Js]  #; gamma = self.gamma
    tail_prob = pi_0 / (1 - rho) * \
        (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
        (1 - (_s*mu - lambda_) / (_s*mu + gamma))**(t*gamma)
    return tail_prob


def V_to_memory(_s, i, s_star):  # _s Extra
    """
    Calculate V for a single queue for all x = -s, ..., 0, ..., D.

    Only call this function once per class for effeciency.
    Handles s > s_star
    (if s > s_star, V(x)=0 for x = -s, ..., 0. As those states are
     never visited.)
    """
    lambda_ = lambda_V[i]; mu = mu_V[i]; t = t_V[i]; weight = weight_V[i]
    # gamma = self.gamma
    a = lambda_ / mu; rho = a / _s

    # Extra, calculate once vectorized, use pi_0[i], g[i], ...
    pi_0 = get_pi_0(_s, rho, i=i)
    tail_prob = get_tail_prob(_s, rho, pi_0, i=i)
    # Scale to get average reward
    g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

    V = np.zeros(_s+1+D)  # V(-s) = 0, reference state
    if(_s <= s_star[i]):  # Extra
        x = np.arange(-_s+1, 1)
        V_x_le_0 = lambda y: (1 - (y/a)**(x+_s)) / (1 - y/a) * np.exp(a-y)
        V[x+_s] = g/lambda_ * quad_vec(V_x_le_0, a, np.inf)[0]
    x = np.array(range(1, D+1))
    tmp_frac = (_s*mu + gamma) / (lambda_ + gamma)
    V[x+_s] = V[_s] + g / (gamma*_s*mu*(1 - rho)**2) * \
        (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x -
         (lambda_ + gamma))
    _sum = np.exp(a) / a**(_s-1) * gamma_fun(_s) * reg_up_inc_gamma(_s, a)
    V[x+_s] += g / (lambda_*gamma*(rho-1)) * \
        (lambda_ - gamma - gamma/rho * _sum) * \
        (-rho + (lambda_ + gamma) / (_s*mu + gamma) * tmp_frac**x)
    # -1_{x > t*gamma}[...]
    alpha = np.floor(t*gamma+1).astype(int)
    x = np.array(range(alpha, D+1))
    V[x+_s] -= weight/(gamma * (1 - rho)**2) * \
        ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) +
         (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
    return V


def trans_prob():
    """
    Precalculate P_xy.

    Jump from x>0 to y. For index convenience P_0y=0
    """
    P_xy = np.zeros((J, D+1, D+1))  # for class i in J, p_{x,y}
    A = np.indices((D+1, D+1))  # x=A[0], x=A[1]
    for i in range(J):
        P_xy[i, 1:, 0] = (gamma / (lambda_V[i] + gamma))**A[0, 1:D+1, 0]
        P_xy[i, 1:, 1:] = (gamma / (lambda_V[i] + gamma)) ** \
            (A[0, 1:, 1:] - A[1, 1:, 1:]) * lambda_V[i] / (lambda_V[i]+gamma)
    P_xy = np.tril(P_xy, 0)
    return P_xy


P_xy = trans_prob()

i = 0
s = s_star[i]
lambda_ = lambda_V[i]
mu = mu_V[i]
rho = lambda_ / (s*mu)
t = t_V[i]
tau = tau_V[i]
weight = weight_V[i]

V = V_to_memory(s_star[i], i, s_star)
x = np.append(np.array(range(-s_star[i], 0)), np.array(range(D+1)))
# np.stack((x, V), axis=1)

pi_0 = get_pi_0(s, rho, i=i)
tail_prob = get_tail_prob(s, rho, pi_0, i=i)
# Scale to get average reward
g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

Equality_R = np.zeros([s+1+D, 4])
Equality_R[:, 0] = x
Equality_R[:, 3] = V
LHS = g + tau*V[x+s]
RHS = np.zeros(s+1+D)
# x == -s
RHS[0] = lambda_*V[1] + (tau-lambda_)*V[0]
# -s < x <= 0
y = x[np.logical_and(x > -s, x <= 0)]
RHS[y+s] = lambda_*V[y+s+1] + (y+s)*mu*V[y+s-1] + \
    (tau-lambda_-(y+s)*mu)*V[y+s]
# x>=1
RHS[s+1:s+1+D-1] = gamma*V[s+2:s+1+D] + \
    s*mu*np.sum(P_xy[i, 1:D+1-1, :1+D]*V[s:s+1+D], 1) + \
    (tau-gamma-s*mu)*V[s+1:s+1+D-1]
alpha = np.floor(t*gamma+1).astype(int)
x_alpha = np.array(range(s+alpha, s+1+D-1))
RHS[x_alpha] += weight*s*mu*np.ones(D+1-1-alpha)

Equality_R[:, 1] = LHS
Equality_R[:, 2] = RHS

dec = 4
print((np.around(Equality_R[:s+D-1, 1], dec) ==
       np.around(Equality_R[:s+D-1, 2], dec)).all())

# x=20
# tmp_sum = 0
# for y in range(0,x+1):
#     if(y==0):
#         p_xy = (gamma / (lambda_ + gamma))**x
#     else:
#         p_xy = (gamma / (lambda_ + gamma))**(x-y) * \
#             lambda_ / (lambda_ + gamma)
#     tmp_sum += p_xy
#     print(p_xy)
# print(tmp_sum)

# 3.8189796030208895e-05
# 2.288400626172035e-05
# 2.2884023740430977e-05
# True
# V(-3) 0.0029806440921911347
# V(-2) 0.009631313491753888
# V(0) 0.15609478643644725
# V(1) 0.7553072270736971
# V(2) 0.8140540024637064
# V(20) 1.8250322080123087

###############################################################################
###############################################################################
