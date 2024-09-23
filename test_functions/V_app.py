"""
Description.

V_f is vectorized over all x (and not over all queues)
Quad_vec limits vectorizing over queues as the limits of the integration
cannot be vectors. Additionally, V then is a matrix with specific dimensions.
It is easier (and more readable and understandable) when we loop over J.

V_app does not need V(x) for x<0. Therefore, these are left out 
(also easier in light of problems with fractional s)

# -s <= x < 0,
g + tau*V(x) = lmbda * (r + V(x+1)) + (x + S)*mu*V(max(x-1,0)) + \
                (tau - lmbda - (x+S)*mu)*V(x)
note V[max(x+S-1,0)] maps V(-s-1) -> V(-s)=0
Moreover, when x=-s then (x+S)*mu=0
"""

import numpy as np
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
from utils import TimeConstraintEDs as Env, OneStepPolicyImprovement as learner

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(42)
env = Env(J=1, S=3, load=0.75, gamma=20., D=100, P=1e3, e=1e-4)
s = env.s_star
S = np.ceil(s).astype(int)
lab = env.lab
mu = env.mu

# -------------------- With x<0 -------------------------
v_app = np.zeros(env.dim)
v = np.zeros(env.J, max(s) + env.D + 1)
for i in np.arange(env.J):
    v[i, :(s[i] + env.D + 1)] = learner.get_v_app_i(env, i)
    for x in np.arange(env.D + 1):
        states = [slice(None)] * (env.J * 2)
        states[i] = x
        v_app[tuple(states)] += v[i, x]
        
for i in np.arange(env.J):
    x = np.arange(-S[i], env.D+1)
    LHS = env.g[i] + env.tau[i] * v[i, x + S[i]]
    RHS = np.zeros(S[i] + env.D + 1)
    # -s <= x < 0
    x = np.arange(-S[i], 0)
    RHS[x + S[i]] = (lab[i] * (env.r[i] + v[i, x + S[i] + 1])
                     + (x + s[i]) * mu[i] * v[i, np.maximum(x + S[i] - 1, 0)]
                     + (env.tau[i] - lab[i] - (x + s[i]) * mu[i])
                     * v[i, x + S[i]])
    x = 0
    RHS[x + S[i]] = (lab[i] * v[i, x + S[i] + 1]
                     + (x + S[i]) * mu * v[i, np.maximum(x + S[i] - 1, 0)]
                     + (env.tau[i] - lab[i] - (x + s[i]) * mu) * v[i, x + S[i]])
    # x >= 1
    x = np.arange(1, env.D)
    RHS[x + S[i]] = (env.gamma * V[i, x + S[i] + 1]
                     + s[i] * mu[i] * (env.r[i] + np.sum(
                env.P_xy[i, 1:env.D, :env.D]
                * v[i, S[i]:env.D + S[i]], 1))
                     + (env.tau[i] - env.gamma - s[i] * mu[i])
                     * V[i, x + S[i]])
    x = np.arange(env.t[i] * env.gamma + 1, env.D).astype(int)  # x>t*gamma
    RHS[x + S[i]] -= s[i] * mu[i] * env.c[i]
    
    print("pi_0: ", env.pi_0[i])
    print("tail_prob: ", env.tail_prob[i])
    print("g: ", env.g[i])
    print("V", v[i,])
    dec = 5
    x = np.arange(-env.S[i], env.env.D)
    print("gamma*t: ", env.gamma * env.t[i])
    print("LHS==RHS? ",
          (np.around(LHS[x + S[i]], dec)
           == np.around(RHS[x + S[i]], dec)).all())
    print("x, LHS, RHS, V: \n",
          np.c_[np.arange(-S[i], env.D+1), LHS, RHS, v[i,]])

# Policy improvement to get policy
# Evaluate policy to get g (and V)
# plot g and V
