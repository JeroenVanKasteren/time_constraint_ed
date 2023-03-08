"""
Description.

# -s <= x < 0,
g + tau*V(x) = lmbda * (r + V(x+1)) + (x + S)*mu*V(max(x-1,0)) + \
                (tau - lmbda - (x+S)*mu)*V(x)
note V[max(x+S-1,0)] maps V(-s-1) -> V(-s)=0, therefore term is ignored
Moreover, when x=-s then (x+S)*mu=0
"""

import numpy as np
from numpy import array, exp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

from OtherTests.init import Env

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(42)
env = Env(J=1, S=1, load=0.75, gamma=20., D=50, P=1000, e=1e-4)
# env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
#           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100)

S = env.S
D = env.D
lab = env.lab
mu = env.mu
gamma = env.gamma
rho = env.rho
a = env.a
r = env.r

def V_f(env, g):
    """Calculate V for a single queue."""
    V = np.zeros(S + env.D + 1)

    x = np.arange(-S+1, +1)  # V(x) for x<=0, with V(-s)=0
    v_x_le_0 = lambda y: (1 - (y / a)**(x + S)) / (1 - y/a) * exp(a - y)
    V[x+S] = (g - lab*r) / lab * quad_vec(v_x_le_0, a, np.inf)[0]

    frac = (S * mu + gamma) / (lab + gamma)
    trm = exp(a) / a ** (S - 1) * gamma_fun(S) * reg_up_inc_gamma(S, a)
    x = np.arange(1, env.D + 1)  # V(x) for x>0
    # x = np.arange(0, env.D + 1)  # V(x) for x>=0
    V[x+S] = (V[S] + (S*mu*r - g) / (gamma*S*mu*(1 - rho)**2)
              * (lab + gamma - lab*x*(rho-1) - (lab + gamma)*frac**x)
              + 1/(gamma*(rho-1))
              * (g - S*mu*env.r - gamma/lab * (g + (g - lab*env.r)/rho * trm))
              * (-rho + frac**(x-1)))
    # 1 if x=0
    # V[S] += (1 / (gamma * (rho - 1))
    #         * (g - S * mu * env.r - gamma / lab * (g + (g - lab*r) / rho*trm))
    #          * (gamma * (rho - 1) / (S * mu + gamma)))

    # -1_{x > gamma*t}[...]
    x = np.arange(gamma*env.t+1, env.D+1).astype(int)
    V[x+S] -= env.c / (gamma * (1 - rho)**2) * \
        (lab + gamma - lab*(x - gamma*env.t - 1) * (rho - 1) -
         (lab + gamma) * frac**(x-gamma*env.t-1))
    return V


pi_0 = env.get_pi_0(env.s_star, env.rho)
tail_prob = env.get_tail_prob(env.S, env.rho, pi_0, env.gamma*env.t)
g = (env.r - env.c*tail_prob) * (env.lab + pi_0*env.lab**2/env.gamma)
P_xy = env.P_xy[0]  # [0]
V = V_f(env, g)

x = np.arange(-S, D+1)
LHS = g + env.tau * V[x+S]
RHS = np.zeros(S + D + 1)
x = np.arange(-S, 0)  # -s <= x < 0
RHS[x+S] = (lab*(env.r+V[x+S+1]) + (x + S)*mu*V[np.maximum(x+S-1, 0)]
            + (env.tau - lab - (x+S)*mu)*V[x+S])
x = 0  # x = 0
RHS[x+S] = (lab*V[x+S+1] + (x + S)*mu*V[np.maximum(x+S-1, 0)]
            + (env.tau - lab - (x + S) * mu) * V[x+S])
x = np.arange(1, D)  # x>=1
RHS[x+S] = (gamma*V[x+S+1]
            + S*mu*(env.r + np.sum(P_xy[1:D, :D]*V[S:D+S], 1))
            + (env.tau - gamma - S*mu)*V[x+S])
x = np.arange(env.t*gamma+1, D).astype(int)  # x>t*gamma
RHS[x+S] -= S*mu * env.c

print("V", V)
dec = 5
x = np.arange(-env.S, env.D)
print("gamma*t: ", env.gamma*env.t)
print("x, LHS, RHS, V: \n", np.c_[np.arange(-env.S, env.D+1), LHS, RHS, V])
print("LHS==RHS? ", (np.around(LHS[x+S], dec) ==
                     np.around(RHS[x+S], dec)).all())
print("pi_0: ", pi_0)
print("tail_prob: ", tail_prob)
print("g: ", g)
# print("g*tau: ", g*env.tau)

# print(env.P_xy)
# V_single_queue = V
# V_single_queue[4:] - V[:,4]
