"""
Sandbox Value Iteration
"""

import numpy as np
import numba as nb
from numba import types as tp
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

pi_learner = PolicyIteration()
env = Env(J=1, S=1, gamma=5, D=15, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=1, print_modulo=1,
          max_iter=10)

DICT_TYPE_I0 = tp.DictType(tp.unicode_type, tp.i8)  # ints
DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F0 = tp.DictType(tp.unicode_type, tp.f8)  # float 1D vector
DICT_TYPE_F1 = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector


# @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.f8, DICT_TYPE_I0,
#                   DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
#          parallel=True, error_model='numpy')
def init_w(V, W, gamma, d_i0, d_i1, d_i2, d_f1, P_xy):
    """W given policy."""
    r, c, t, J, P = d_f1['r'], d_f1['c'], d_f1['t'], d_i0['J'], d_i0['P']
    S, D = d_i0['S'], d_i0['D']
    sizes_x, sizes_s = d_i1['sizes_i'][1:J + 1], d_i1['sizes_i'][J + 1:J*2 + 1]
    sizes_x_n, sizes_s_n = d_i1['sizes'][0:J], d_i1['sizes'][J:J * 2]
    for x_i in nb.prange(len(d_i2['x'])):  # nb.prange
        for s_i in nb.prange(len(d_i2['s_valid'])):
            x = d_i2['x'][x_i]
            s = d_i2['s_valid'][s_i]
            state = np.sum(x * sizes_x_n + s * sizes_s_n)
            over_target = 0
            for k in range(J):
                over_target += x[k] > t[k] * gamma
            for i in range(J + 1):
                state_i = (i * d_i1['sizes_i'][0]
                           + np.sum(x * sizes_x + s * sizes_s))
                if (i < J) and (x[i] < D):
                    W[state_i] = V[state + sizes_x_n[i]]
                else:
                    W[state_i] = V[state]
                if sum(s) < S:
                    if over_target == J:
                        W[state_i] -= P
                    for j in range(J):
                        if (x[j] > 0) or (j == i):
                            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                            i_not_admitted = 0
                            if (i < J) and (i != j) and (x[i] < D):
                                i_not_admitted = sizes_x_n[i]
                            for y in range(x[j] + 1):
                                next_state = (np.sum(x * sizes_x_n
                                                     + s * sizes_s_n)
                                              - (x[j] - y) * sizes_x_n[j]
                                              + i_not_admitted
                                              + sizes_s_n[j])
                                w += P_xy[j, x[j], y] * V[next_state]
                            if w > W[state_i]:
                                W[state_i] = w
    return W


# @nb.njit(nb.types.Tuple((tp.f4[:], tp.f8, tp.b1, tp.i8))(
#     tp.f4[:], tp.f4[:], tp.f4[:], tp.f8, DICT_TYPE_I0, DICT_TYPE_I1,
#     DICT_TYPE_I2, DICT_TYPE_F0, DICT_TYPE_F1, tp.f8[:, :, :]),
#          parallel=True, error_model='numpy')
def value_iteration(V, W, delta, gamma, d_i0, d_i1, d_i2, d_f0, d_f1, P_xy):
    """W given policy."""
    J = d_i0['J']
    sizes_x = d_i1['sizes_i'][1:J + 1]
    sizes_s = d_i1['sizes_i'][J + 1:J*2 + 1]
    sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i1['sizes'][J:J * 2]
    lab, mu = d_f1['lab'], d_f1['mu']
    V_t = tp.float32(d_f0['tau']) * V  # Copies V
    time = 0
    converged = False
    count = tp.int64(0)
    g = float(0)
    delta_max = np.float32(-np.inf)
    delta_min = np.float32(np.inf)
    while ((not converged) and (count < d_f0['max_iter'])
           and (time < d_f0['max_time'])):
        W = init_w(V, W, gamma, d_i0, d_i1, d_i2, d_f1, P_xy)
        for x_i in range(len(d_i2['x'])):  # nb.prange
            for s_i in range(len(d_i2['s_valid'])):
                x, s = d_i2['x'][x_i], d_i2['s_valid'][s_i]
                state = np.sum(x * sizes_x_n + s * sizes_s_n)
                for i in range(J):
                    state_i = (i * d_i1['sizes_i'][0]
                               + np.sum(x * sizes_x + s * sizes_s))
                    if x[i] == 0:
                        V_t[state] += lab[i] * (W[state_i] - V[state])
                    else:
                        V_t[state] += gamma * (W[state_i] - V[state])
                    if s[i] > 0:
                        state_i = (J * d_i1['sizes_i'][0]
                                   + np.sum(x * sizes_x + s * sizes_s))
                        state_i -= sizes_s[i]
                        V_t[state] += (s[i] * mu[i] * (W[state_i] - V[state]))
                V_t[state] /= d_f0['tau']
                delta[state] = abs(V_t[state] - V[state])
        if count % d_i0['convergence_check'] == 0:
            delta_max = delta[0]
            delta_min = delta[0]
            for x_i in range(len(d_i2['x'])):
                if delta_max - delta_min > d_f0['e']:
                    break
                for s_i in range(len(d_i2['s_valid'])):
                    if delta_max - delta_min > d_f0['e']:
                        break
                    x, s = d_i2['x'][x_i], d_i2['s_valid'][s_i]
                    state = np.sum(x * sizes_x_n + s * sizes_s_n)
                    if delta[state] > delta_max:
                        delta_max = delta[state]
                    if delta[state] < delta_min:
                        delta_min = delta[state]
            converged = delta_max - delta_min < d_f0['e']
            g = (delta_max + delta_min) / 2 * d_f0['tau']
            # with objmode(time1='f8'):
            #     time = clock() - d_f0['start_time']
        if count % d_i0['print_modulo'] == 0:
            print('iter:', count,
                  'delta:', delta_max - delta_min,
                  '[d_min, d_max] : [', delta_min, ',', delta_max,
                  '] g:', g)
        V = V_t - V_t[0]
        count += 1
    return V, g, converged, count

# Value Iteration
name = 'Value Iteration'
V = np.zeros(env.size, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.size_i, dtype=np.float32)
delta = np.ones(env.size, dtype=np.float32) * np.inf

start_VI = clock()
V, g, converged, count = value_iteration(V, W, delta, env.gamma, env.d_i0,
                                         env.d_i1, env.d_i2, env.d_f0, env.d_f1,
                                         env.P_xy)
env.time_print(clock()-start_VI)
print(f"g %.3f, converged {converged}, count {count} " % g)
