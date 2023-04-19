"""
Sandbox Value Iteration
"""

import numpy as np
import numba as nb
from numba import types as tp
from FunctionTests.Sandbox_PI import init_w, init_pi, get_v, policy_improvement
from OtherTests.init import Env
from Plotting import plot_pi, plot_v

np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(J=2, S=2, gamma=30., P=1e3, e=1e-5, trace=True,
          lab=np.array([0.6726, 0.1794]), mu=np.array([0.8169, 0.2651]),
          convergence_check=10, print_modulo=10)
# env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
#           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100,
#           max_iter=5)

DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w(V, W, J, D, gamma, d_i1, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i1['sizes_i'][1:J + 1]
    sizes_s = d_i1['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i1['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    for x_i in nb.prange(len(d_i2['x'])):
        for s_i in nb.prange(len(d_i2['s'])):
            for i in nb.prange(J + 1):
                x = d_i2['x'][x_i]
                s = d_i2['s'][s_i]
                state = i * d_i1['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
                for j in range(J):
                    if (x[j] > 0) or (j == i):
                        w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        i_not_admitted = 0
                        if (i < J) and (i != j) and (x[i] < D):
                            i_not_admitted = sizes_x_n[i]
                        for y in range(x[j] + 1):
                            next_state = (np.sum(x*sizes_x_n + s*sizes_s_n)
                                          - (x[j] - y) * sizes_x_n[j]
                                          + i_not_admitted
                                          + sizes_s_n[j])
                            w += P_xy[j, x[j], y] * V[next_state]
                        if w > W[state]:
                            W[state] = w
    return W


# Value Iteration
name = 'Value Iteration'
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = init_pi(env)

count = 0
converged = False

env.timer(True, name, env.trace)
while not converged:  # Update each state.
    W = init_w(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    W = get_w(V, W, env.J, env.D, env.gamma, env.d_i1, env.d_i2, env.d_f,
              env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
    V_t = get_v(env, V, W)
    if count % env.convergence_check == 0:
        converged, g = env.convergence(V_t, V, count, name)
    V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
    if count > env.max_iter:
        break
    count += 1
env.timer(False, name, env.trace)

# Determine policy via Policy Improvement.
W = init_w(env, V, W)
V = V.reshape(env.size)
W = W.reshape(env.size_i)
Pi = Pi.reshape(env.size_i)
Pi, _ = policy_improvement(V, W, Pi, env.J, env.D, env.gamma,
                           env.KEEP_IDLE, env.d_i1, env.d_i2, env.d_f,
                           env.P_xy)
V = V.reshape(env.dim)
Pi = Pi.reshape(env.dim_i)

print("V", V)
print("Pi", Pi)
print("g", g)

if env.J > 1:
    plot_pi(env, env, Pi, zero_state=True)
    plot_pi(env, env, Pi, zero_state=False)
for i in range(env.J):
    plot_pi(env, env, Pi, zero_state=True, i=i)
    plot_pi(env, env, Pi, zero_state=True, i=i, smu=True)
    plot_v(env, V, zero_state=True, i=i)
