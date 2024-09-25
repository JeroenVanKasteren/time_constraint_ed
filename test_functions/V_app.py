"""
Description.

V_f is vectorized over all x (and not over all queues)
Quad_vec limits vectorizing over queues as the limits of the integration
cannot be vectors. Additionally, V then is a matrix with specific dimensions.
It is easier (and more readable and understandable) when we loop over J.

# -s <= x < 0,
g + tau*V(x) = lmbda * (r + V(x+1)) + (x + S)*mu*V(max(x-1,0)) + \
                (tau - lmbda - (x+S)*mu)*V(x)
note V[max(x+S-1,0)] maps V(-s-1) -> V(-s)=0
Moreover, when x=-s then (x+S)*mu=0
"""

import numpy as np
from utils import (TimeConstraintEDs as Env,
                   OneStepPolicyImprovement as OSPI)

np.set_printoptions(precision=4, linewidth=150, suppress=True)
np.random.seed(42)
tolerance = 1e-4

# env = Env(J=1, S=3, load=0.75, gamma=20., D=100, P=1e3, e=1e-4)
env = Env(J=2, S=2, gamma=20., P=1e3, e=1e-5,
          lab=np.array([0.6726, 0.1794]), mu=np.array([0.8169, 0.2651]))
s = env.s_star
S = np.ceil(s).astype(int)
lab = env.lab
mu = env.mu
learner = OSPI()
tau = np.maximum(lab, env.gamma) + s * mu

# ------------------ v_app_i -----------------
for i in np.arange(env.J):
    v = learner.get_v_app_i(env, i)
    x = np.arange(-S[i], env.D + 1)
    LHS = env.g[i] + tau[i] * v[x + S[i]]
    RHS = np.zeros(S[i] + env.D + 1)
    # -s <= x < 0, note V(-s)=0
    x = np.arange(-S[i], 0)
    RHS[x + S[i]] = (lab[i] * (env.r[i] + v[x + S[i] + 1])
                     + (x + S[i]) * mu[i] * v[np.maximum(x + S[i] - 1, 0)]
                     + (tau[i] - lab[i] - (x + S[i]) * mu[i]) * v[x + S[i]])
    x = 0
    RHS[x + S[i]] +=  lab[i] * env.r[i]
    # x >= 1
    x = np.arange(1, env.D)
    RHS[x + S[i]] = (env.gamma * v[x + S[i] + 1]
                     + s[i] * mu[i]
                     * (env.r[i] + np.sum(env.p_xy[i, 1:env.D, :env.D]
                                          * v[S[i]:env.D + S[i]], 1))
                     + (tau[i] - env.gamma - s[i] * mu[i]) * v[x + S[i]])
    # x > t * gamma
    x = np.arange(env.t[i] * env.gamma + 1, env.D).astype(int)
    RHS[x + S[i]] -= s[i] * mu[i] * env.c[i]
    # x = D, Note V[D+1] does not exist
    RHS[env.D + S[i]] = np.nan
    print(f'g: {env.g[i]:0.4f}, '
          f'pi_0: {env.pi_0[i]:0.4f}, '
          f'tail_prob: {env.tail_prob[i]:0.4f}')
    x = np.arange(-S[i], env.D)
    print("gamma*t: ", env.gamma * env.t[i])
    print("LHS==RHS? ",
          np.allclose(LHS[x + S[i]], RHS[x + S[i]], atol=tolerance))
    print("x, LHS, RHS, V: \n",
          np.c_[np.arange(-S[i], env.D + 1), LHS, RHS, v])
# len(np.arange(-S[i], env.D + 1)  # , LHS, RHS, v)
# len(v)
# -------------------- With x<0 -------------------------
v_app_old = learner.get_v_app_old(env)
v_app_base = learner.get_v_app_base(env)
v_app_lin = learner.get_v_app_lin(env, type='linear')
v_app_abs = learner.get_v_app_lin(env, type='abs')
        
# Policy improvement to get policy
# Evaluate policy to get g (and V)
# plot g and V
