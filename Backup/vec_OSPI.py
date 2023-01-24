# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:52:25 2020

@author: Jeroen
"""

import numpy as np
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

J = 3
lambda_V = np.array([1, 1.5, 0.5])
mu_V = np.array([0.5, 1, 0.5])
s_V = 10
print(sum(lambda_V/mu_V)/s_V)  # Total system load < 1
print(lambda_V/mu_V)  # Used to estimate s_star
s_star = np.array([4, 2, 2])
t_V = np.array([1/5]*J)
gamma = 5
tau_V = np.maximum(lambda_V, gamma) + s_star*mu_V
D = 20
weight_V = np.array([2, 3, 1])


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
V_app = self.approx_V(s_star)
# Calculate per state best action (policy) by one-step lookahead
it = np.nditer(self.policy, flags=['multi_index']); pointer = -1
while not it.finished:
    multi_state = it.multi_index; it.iternext()
    if(multi_state[0] > pointer):
        print(multi_state); pointer += 1
    states = np.array(multi_state).reshape(J, 2)
    # If no server free or no one waiting, no decision needed.
    if((sum(states[:,1]) >= self.s) | (sum(states[:,0]) == 0)):
        self.policy[multi_state] = J
    else:
        action_values = np.zeros(J+1)
        for i in range(J):
            x_i = multi_state[i*2]
            if(x_i > 0):  # If someone of class i waiting
                next_state = list(multi_state)
                next_state[i*2] = list(range(x_i, -1, -1))
                next_state[i*2+1] += 1  # s
                # Missing code that sum of s < s
                action_values[i] = self.env.c(x_i, i) + \
                    np.sum(P_xy[i, x_i, range(x_i+1)] * \
                           V_app[tuple(next_state)])
            else:  # Else no one of class i to take into service
                action_values[i] = np.inf
        action_values[J] = V_app[multi_state]  # do nothing