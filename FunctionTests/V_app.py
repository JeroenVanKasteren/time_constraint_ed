"""
Description.

V_f is vectorized over all x (and not over all queues)
Quad_vec limits vectorizeing over queues as the limits of the integration 
cannot be vectors. Additionally, V then is a matrix with specific dimensions.
It is easier (and more readible and understandeble) when we loop over J.

V_app does not need V(x) for x<0. Therefore, these are left out 
(also easier in light of problems with fractional s)
"""

PATH = (r"D:\Programs\Surfdrive\Surfdrive\VU\Promovendus"
        r"\Time constraints in emergency departments\Code")
# PATH = (r"C:\Users\jkn354\surfdrive\VU\Promovendus"
#         r"\Time constraints in emergency departments\Code")

import os
os.chdir(PATH+"\Other")
from init import Env

import numpy as np
from numpy import array, arange, zeros, around, exp

from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec

np.set_printoptions(precision=3, linewidth=150, suppress=True)
env = Env(J=3, S=10,
          mu=np.array([2, 2, 2]),
          lmbda=np.array([1.5, 1, 1.5]),
          t=array([3]*3),
          r=array([1]*3), c=np.array([1, 1, 1]),
          gamma=2, D=15,
          trace=True, time_check=False)

# -------------------- Without x<0 -------------------------

def V_app_f(env, i):
    """Calculate V for a single queue."""
    s=env.s_star[i]; lmbda=env.lmbda[i]; mu=env.mu[i]; rho=env.rho[i]; 
    a=env.a[i]; r=env.r[i]; c=env.c[i]; t=env.t[i]; g=env.g[i]
    V_i = zeros(env.D + 1)

    # V(x) for x<=0, with V(-s)=0
    V_x_le_0 = lambda y: (1-(y/a)**(s))/(1-y/a)*exp(a-y)
    V_i[0] = (g - lmbda*r)/lmbda * quad_vec(V_x_le_0, a, np.inf)[0]

    # V(x) for x>0
    x = arange(1, env.D+1) 
    frac = (s*mu + env.gamma) / (lmbda + env.gamma)
    trm = exp(a)/a**(s-1) * gamma_fun(s)*reg_up_inc_gamma(s,a)
    V_i[x] = V_i[0] + (s*mu*r - g) / (env.gamma*s*mu*(1 - rho)**2) * \
        (lmbda + env.gamma - lmbda*x*(rho-1) - (lmbda+env.gamma)*frac**x) + \
            1/(env.gamma*(rho-1)) * (g - s*mu*r - env.gamma/lmbda *(
                g + (g-lmbda*r)/rho * trm)) * (-rho + frac**(x-1))
    # -1_{x > gamma*t}[...]
    x = arange(env.gamma*t + 1, env.D + 1).astype(int)
    V_i[x] -= c / (env.gamma * (1 - rho)**2) * \
        (lmbda + env.gamma - lmbda*(x - env.gamma*t - 1) * \
         (rho - 1) - (lmbda + env.gamma) * frac**(x-env.gamma*t-1))
    return V_i
        
V_app = zeros(env.dim)
V = zeros((env.J, env.D+1))
for i in arange(env.J):
    V[i,] = V_app_f(env, i)
    for x in arange(env.D + 1):
        states = [slice(None)]*(env.J*2)
        states[i] = x
        V_app[tuple(states)] += V[i, x]

for i in arange(env.J):
    s=env.s_star[i]; lmbda=env.lmbda[i]; mu=env.mu[i]; rho=env.rho[i]; 
    a=env.a[i]; r=env.r[i]; c=env.c[i]; t=env.t[i]; g=env.g[i]
    tau = s*mu + np.maximum(lmbda, env.gamma)

    x = arange(env.D+1)
    LHS = g + tau*V[i,x]
    RHS = np.zeros(env.D+1)
    x = arange(1,env.D)  # x>=1
    RHS[x] = env.gamma*V[i,x+1] + \
        s*mu*(r + np.sum(env.P_xy[i, 1:env.D,:env.D]*V[i,0:env.D],1)) + \
            (tau - env.gamma - s*mu)*V[i,x]
    x = arange(t*env.gamma+1,env.D).astype(int)  # x>t*gamma
    RHS[x] -= s*mu*c
    print("tail_prob: ", env.tail_prob)
    print("g: ", env.g)
    print("V", V[i,:])
    dec = 5
    x = arange(1, env.D)
    print("gamma*t: ", env.gamma*t)
    print("LHS==RHS? ", (around(LHS[x], dec) == around(RHS[x], dec)).all())
    print("x, LHS, RHS, V: \n", np.c_[arange(env.D+1), LHS, RHS, V[i,:]])

# -------------------- With x<0 -------------------------

# def V_f(env, i):
#     """Calculate V for a single queue."""
#     s=env.s_star[i]; lmbda=env.lmbda[i]; mu=env.mu[i]; rho=env.rho[i]; 
#     a=env.a[i]; r=env.r[i]; c=env.c[i]; t=env.t[i]; g=env.g[i]
#     s_ = int(s)
#     V_i = zeros(s_ + env.D + 1)
    
#     x = arange(-s_+1,0+1)  # V(x) for x<=0, with V(-s)=0
#     V_x_le_0 = lambda y: (1-(y/a)**(x+s))/(1-y/a)*exp(a-y)
#     V_i[x+s_] = (g - lmbda*r)/lmbda * quad_vec(V_x_le_0, a, np.inf)[0]
    
#     x = arange(1, env.D+1)  # V(x) for x>0
#     frac = (s*mu + env.gamma) / (lmbda + env.gamma)
#     trm = exp(a)/a**(s-1) * gamma_fun(s)*reg_up_inc_gamma(s,a)
#     V_i[x+s_] = V_i[s_] + (s*mu*r - g) / (env.gamma*s*mu*(1 - rho)**2) * \
#         (lmbda + env.gamma - lmbda*x*(rho-1) - (lmbda+env.gamma)*frac**x) + \
#             1/(env.gamma*(rho-1)) * (g - s*mu*r - env.gamma/lmbda *(
#                 g + (g-lmbda*r)/rho * trm)) * (-rho + frac**(x-1))
#     # -1_{x > gamma*t}[...]
#     x = arange(env.gamma*t + 1, env.D + 1).astype(int)
#     V_i[x+s_] -= c / (env.gamma * (1 - rho)**2) * \
#         (lmbda + env.gamma - lmbda*(x - env.gamma*t - 1) * \
#          (rho - 1) - (lmbda + env.gamma) * frac**(x-env.gamma*t-1))
#     return V_i

# V_app = zeros(env.dim)
# V = zeros((env.J, int(max(env.s_star)+env.D+1)))
# for i in arange(env.J):
#     V[i,:int(env.s_star[i]+env.D+1)] = V_f(env, i)
#     for x in arange(env.D + 1):
#         states = [slice(None)]*(env.J*2)
#         states[i] = x
#         V_app[tuple(states)] += V[i, x]
        
# for i in arange(env.J):
#     s=env.s_star[i]; lmbda=env.lmbda[i]; mu=env.mu[i]; rho=env.rho[i]; 
#     tau=env.tau[i]; a=env.a[i]; r=env.r[i]; c=env.c[i]; t=env.t[i]; g=env.g[i]
#     s_ = int(s)
    
#     x = arange(-s_,env.D+1)
#     LHS = g + env.tau*V[i,x+s_]
#     RHS = np.zeros(s_+env.D+1)
#     x = arange(-s_,0)  # -s <= x < 0
#     RHS[x+s_] = lmbda*(env.r+V[i,x+s_+1]) + (x+s)*mu*V[i,maximum(x+s_-1,0)]+ \
#         (env.tau - lmbda - (x+s)*mu)*V[i,x+s_]
#     x = 0  # x = 0
#     RHS[x+s_] = lmbda*V[i,x+s_+1] + (x + s)*mu*V[i,maximum(x+s_-1,0)] + \
#         (tau - lmbda - (x+s)*mu)*V[i,x+s_]
#     x = arange(1,env.D)  # x>=1
#     RHS[x+s_] = env.gamma*V[i,x+s_+1] + \
#         s*mu*(r + np.sum(env.P_xy[i,1:env.D,:env.D]*V[i,s_:env.D+s_],1)) + \
#             (tau - env.gamma - s_*mu)*V[i,x+s_]
#     x = arange(t*env.gamma+1,env.D).astype(int)  # x>t*gamma
#     RHS[x+s_] -= s*mu*c
    
#     print("pi_0: ", env.pi_0)
#     print("tail_prob: ", env.tail_prob)
#     print("g: ", env.g)
#     print("V", V[i,])
#     dec = 5
#     x = arange(-s_, env.env.D)
#     print("gamma*t: ", env.gamma*t)
#     print("LHS==RHS? ", (around(LHS[x+s_], dec) == around(RHS[x+s_], dec)).all())
#     print("x, LHS, RHS, V: \n", np.c_[arange(-s_, env.D+1), LHS, RHS, V[i,]])
