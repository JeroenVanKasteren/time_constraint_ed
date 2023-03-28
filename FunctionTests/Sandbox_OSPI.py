"""
Sandbox One-step policy improvement

Note that V(x) for x<=0 is used to calculate V(0), which is used in V(x) x>0.
"""

import numpy as np
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
from Sandbox_PI import init_w, init_pi, policy_improvement, policy_evaluation
from OtherTests.init import Env
from src.Plotting import plot_pi, plot_v

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(42)
env = Env(J=1, S=2, load=0.75, gamma=20., D=10, P=1e3, e=1e-5, trace=True,
          convergence_check=10, print_modulo=10)
# env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
#           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100,
#           max_iter=5)


def get_v_app_i(env, i):
    """Calculate V for a single queue."""
    s, lab, mu, r = env.s_star[i], env.lab[i], env.mu[i], env.r[i]
    rho, a = env.rho[i], env.a[i]
    g = env.g[i]
    v_i = np.zeros(env.D + 1)

    # V(x) for x<=0, with V(-s)=0
    v_x_le_0 = lambda y: (1 - (y / a) ** s) / (1 - y / a) * np.exp(a - y)
    v_i[0] = (g - lab * r) / lab * quad_vec(v_x_le_0, a, np.inf)[0]
    # V(x) for x>0
    frac = (s * mu + env.gamma) / (lab + env.gamma)
    trm = np.exp(a) / a ** (s - 1) * gamma_fun(s) * reg_up_inc_gamma(s, a)
    x = np.arange(1, env.D + 1)
    v_i[x] = (v_i[0] + (s * mu * r - g) / (env.gamma * s * mu * (1 - rho) ** 2)
              * (lab + env.gamma - lab * x * (rho - 1)
                 - (lab + env.gamma) * frac ** x)
              + 1 / (env.gamma * (rho - 1))
              * (g - s * mu * r - env.gamma / lab * (
                        g + (g - lab * r) / rho * trm))
              * (-rho + frac ** (x - 1)))
    # -1_{x > gamma*t}[...]
    x = np.arange(env.gamma * env.t[i] + 1, env.D + 1).astype(int)
    v_i[x] -= env.c[i] / (env.gamma * (1 - rho) ** 2) * \
              (lab + env.gamma - lab * (x - env.gamma * env.t[i] - 1)
               * (rho - 1) - (lab + env.gamma) * frac ** (
                       x - env.gamma * env.t[i] - 1))
    return v_i


def get_v_app(env):
    """Approximation of value function.

    Create a list V_memory with V_ij(x), i=class, j=#servers for all x.
    Note only j = s*_i, ..., s will be filled, rest zero
    """
    V_app = np.zeros(env.dim, dtype=np.float32)
    V = np.zeros((env.J, env.D + 1))
    for i in range(env.J):
        V[i, ] = get_v_app_i(env, i)
        for x in range(env.D + 1):
            states = [slice(None)] * (env.J * 2)
            states[i] = x
            V_app[tuple(states)] += V[i, x]
    return V_app

# ----------------------- One Step Policy Improvement ----------------------
name = 'One-step Policy Improvement'
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = init_pi(env)
g = 0

env.timer(True, name, env.trace)
V_app = get_v_app(env)
W = init_w(env, V_app, W)
V_app = V_app.reshape(env.size)
W = W.reshape(env.size_i)
Pi = Pi.reshape(env.size_i)
Pi, _ = policy_improvement(V_app, W, Pi, env.J, env.D, env.gamma, env.KEEP_IDLE,
                           env.d_i1, env.d_i2, env.d_f, env.P_xy)
env.timer(False, name, env.trace)

env.timer(True, name, env.trace)
V_app = V_app.reshape(env.dim)
W = W.reshape(env.dim_i)
_, g = policy_evaluation(env, V_app, W, Pi, g, name, count=0)
env.timer(False, name, env.trace)

Pi = Pi.reshape(env.dim_i)

print("V", V_app)
print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_pi(env, env, Pi, zero_state=True)
    plot_pi(env, env, Pi, zero_state=False)
for i in range(env.J):
    plot_pi(env, env, Pi, zero_state=True, i=i)
    plot_pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_v(env, V_app, zero_state=True, i=i)
