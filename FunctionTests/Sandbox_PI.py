"""
Sandbox Policy Iteration
"""

import numpy as np
from numpy import array
import numba as nb
from numba import types as tp
from OtherTests.init import Env
from src.Plotting import plot_pi, plot_v

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(42)
# env = Env(J=2, S=4, load=0.5, gamma=10., D=25, P=1e3, e=1e-4, trace=True,
#           print_modulo=100)
env = Env(J=2, S=3, load=0.75, gamma=5., D=10, P=1e3, e=1e-5, trace=True,
          convergence_check=10, print_modulo=10)
# env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
#           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100,
#           max_iter=5)

DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

d_i1 = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.i4[:])
d_i1['sizes'] = env.sizes
d_i1['sizes_i'] = env.sizes_i
d_i2 = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.i4[:, :])
d_i2['s'] = env.s_states
d_i2['x'] = env.x_states
d_f = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.f8[:])
d_f['t'] = env.t
d_f['c'] = env.c
d_f['r'] = env.r


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w(V, W, Pi, J, D, gamma,
          d_i, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    for s_i in nb.prange(len(d_i2['s'])):
        for x_i in nb.prange(len(d_i2['x'])):
            for i in nb.prange(J + 1):
                x = d_i2['x'][x_i]
                s = d_i2['s'][s_i]
                state = i * d_i['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
                if Pi[state] > 0:
                    j = Pi[state] - 1
                    W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                    i_not_admitted = 0
                    if (i < J) and (i != j) and (x[i] < D):
                        i_not_admitted = sizes_x_n[i]
                    for y in range(x[j] + 1):
                        next_state = (np.sum(x*sizes_x_n + s*sizes_s_n)
                                      - (x[j] - y) * sizes_x_n[j]
                                      + i_not_admitted
                                      + sizes_s_n[j])
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W


def get_v(env, V, W):
    """V_t."""
    states_c = [slice(None)] * (env.J * 2)
    V_t = env.tau * V
    for i in range(env.J):
        states_i = np.append(i, [slice(None)] * (env.J * 2))

        states = states_c.copy()
        next_states = states_i.copy()
        states[i] = 0  # x_i = 0
        next_states[1 + i] = 0
        V_t[tuple(states)] += env.lab[i] * (W[tuple(next_states)]
                                            - V[tuple(states)])

        states = states_c.copy()
        next_states = states_i.copy()
        states[i] = slice(1, env.D + 1)  # 0 < x_i <= D
        next_states[1 + i] = slice(1, env.D + 1)  # 0 < x_i <= D
        V_t[tuple(states)] += env.gamma * (W[tuple(next_states)]
                                           - V[tuple(states)])

        for s_i in range(1, env.S + 1):  # s_i
            states = states_c.copy()
            next_states = states_i.copy()
            states[env.J + i] = s_i
            next_states[0] = env.J
            next_states[1 + env.J + i] = s_i - 1
            V_t[tuple(states)] += s_i * env.mu[i] * (W[tuple(next_states)]
                                                     - V[tuple(states)])
    return V_t / env.tau


@nb.njit(nb.types.Tuple((nb.i4[:], nb.b1))(
    tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8, tp.i8,
    DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
    parallel=True, error_model='numpy')
def policy_improvement(V, W, Pi, J, D, gamma, keep_idle,
                       d_i, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    stable = 0
    for s_i in nb.prange(len(d_i2['s'])):
        for x_i in nb.prange(len(d_i2['x'])):
            for i in nb.prange(J + 1):
                x = d_i2['x'][x_i]
                s = d_i2['s'][s_i]
                state = i * d_i['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
                pi = Pi[state]
                if (np.sum(x) > 0) or (i < J):
                    Pi[state] = keep_idle
                w = W[state]
                for j in range(J):  # j waiting, arrival, or time passing
                    if (x[j] > 0) or (j == i):
                        value = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        i_not_admitted = 0
                        if (i < J) and (i != j) and (x[i] < D):
                            i_not_admitted = sizes_x_n[i]
                        for y in range(x[j] + 1):
                            next_state = (np.sum(x * sizes_x_n + s * sizes_s_n)
                                          - (x[j] - y) * sizes_x_n[j]
                                          + i_not_admitted
                                          + sizes_s_n[j])
                            value += P_xy[j, x[j], y] * V[next_state]
                        if value > w:
                            Pi[state] = j + 1
                            w = value
                if pi != Pi[state]:
                    stable = stable + 1  # binary operation allows reduction
    return Pi, stable == 0


def policy_evaluation(env, V, W, Pi, name, count=0):
    """Policy Evaluation."""
    inner_count = 0
    converged = False
    while not converged:
        W = env.init_w(V, W)
        V = V.reshape(env.size)
        W = W.reshape(env.size_i)
        W = get_w(V, W, Pi, env.J, env.D, env.gamma, d_i1, d_i2, d_f, env.P_xy)
        V = V.reshape(env.dim)
        W = W.reshape(env.dim_i)
        V_t = get_v(env, V, W)
        if inner_count % env.convergence_check == 0:
            converged, g = env.convergence(V_t, V, count, name, j=inner_count)
        V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
        if count > env.max_iter:
            return V, g
        inner_count += 1
    return V, g


# Policy Iteration
name = 'Policy Iteration'
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = env.init_pi()
Pi = Pi.reshape(env.size_i)

count = 0
stable = False

env.timer(True, name, env.trace)
while not stable:
    V, g = policy_evaluation(env, V, W, Pi, 'Policy Evaluation of PI', count)
    W = env.init_w(V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    Pi, stable = policy_improvement(V, W, Pi, env.J, env.D, env.gamma,
                                    env.KEEP_IDLE, d_i1, d_i2, d_f, env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
    if count > env.max_iter:
        break
    count += 1
env.timer(False, name, env.trace)

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
