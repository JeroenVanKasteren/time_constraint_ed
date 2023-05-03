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
env = Env(J=2, S=3, gamma=10, D=30, P=1e3, e=1e-5, seed=42,
          max_time='0-00:05:30', convergence_check=10, print_modulo=100)

DICT_TYPE_I0 = tp.DictType(tp.unicode_type, tp.i4)  # ints
DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F0 = tp.DictType(tp.unicode_type, tp.f8)  # float 1D vector
DICT_TYPE_F1 = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector


@nb.njit(tp.f4(tp.f4[:], tp.i8, tp.i4[:], tp.i4[:], tp.i8, tp.f8,
                  DICT_TYPE_I0, DICT_TYPE_I1, DICT_TYPE_F1, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w_i(V, i, x, s, state, gamma, d_i0, d_i1, d_f1, P_xy):
    J, D, r, c, t = d_i0['J'], d_i0['D'], d_f1['r'], d_f1['c'], d_f1['t']
    sizes_x_n, sizes_s_n = d_i1['sizes'][0:J], d_i1['sizes'][J:J * 2]
    next_state = state + d_i1['sizes'][i] if x[i] < D else state
    w_res = V[next_state]
    if s == d_i0['S']:
        return w_res
    if sum(x > gamma * t) == J:
        w_res -= d_i0['P']
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
            if w > w_res:
                w_res = w
    return w_res

@nb.njit(tp.f4(tp.f4[:], tp.i4[:], tp.i4[:], tp.i8, tp.f8,
                  DICT_TYPE_I0, DICT_TYPE_I1, DICT_TYPE_F1, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w(V, x, s, state, gamma, d_i0, d_i1, d_f1, P_xy):
    J, D, r, c, t = d_i0['J'], d_i0['D'], d_f1['r'], d_f1['c'], d_f1['t']
    sizes_x_n, sizes_s_n = d_i1['sizes'][0:J], d_i1['sizes'][J:J * 2]
    w_res = V[state]
    if s == d_i0['S']:
        return w_res
    if sum(x > gamma * t) == J:
        w_res -= d_i0['P']
    for j in range(J):
        if x[j] > 0:
            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
            for y in range(x[j] + 1):
                next_state = (np.sum(x*sizes_x_n + s*sizes_s_n)
                              - (x[j] - y) * sizes_x_n[j]
                              + sizes_s_n[j])
                w += P_xy[j, x[j], y] * V[next_state]
            if w > w_res:
                w_res = w
    return w_res


# # @staticmethod
# @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
#                   DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
#          parallel=True, error_model='numpy')
# def get_w(V, W, J, D, gamma, d_i1, d_i2, d_f, P_xy):
#     """W given policy."""
#     sizes_x, sizes_s = d_i1['sizes_i'][1:J + 1], d_i1['sizes_i'][J + 1:J*2 + 1]
#     sizes_x_n, sizes_s_n = d_i1['sizes'][0:J], d_i1['sizes'][J:J * 2]
#     r, c, t = d_f['r'], d_f['c'], d_f['t']
#     for x_i in nb.prange(len(d_i2['x'])):
#         for s_i in nb.prange(len(d_i2['s'])):
#             for i in nb.prange(J + 1):
#                 x = d_i2['x'][x_i]
#                 s = d_i2['s'][s_i]
#                 state = np.sum(x * sizes_x_n + s * sizes_s_n)
#                 state_i = i * d_i1['sizes_i'][0] + np.sum(
#                     x * sizes_x + s * sizes_s)
#                 W[state_i] = V[state]
#                 if P > 0:
#                     count = 0
#                     for k in range(J):
#                         count += x[i] > t[i]*gamma
#                     if count == J:  # by definition, sum(s_i) = S
#                         W[state_i] -= P
#                 for j in range(J):
#                     if (x[j] > 0) or (j == i):
#                         w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
#                         i_not_admitted = 0
#                         if (i < J) and (i != j) and (x[i] < D):
#                             i_not_admitted = sizes_x_n[i]
#                         for y in range(x[j] + 1):
#                             next_state = (np.sum(
#                                 x * sizes_x_n + s * sizes_s_n)
#                                           - (x[j] - y) * sizes_x_n[j]
#                                           + i_not_admitted
#                                           + sizes_s_n[j])
#                             w += P_xy[j, x[j], y] * V[next_state]
#                         if w > W[state_i]:
#                             W[state_i] = w
#     return W


@nb.njit(nb.types.Tuple((tp.f4[:], tp.f8, tp.b1, tp.i8))(
    tp.f4[:], tp.f4[:], tp.f8, DICT_TYPE_I0, DICT_TYPE_I1, DICT_TYPE_I2,
    DICT_TYPE_F0, DICT_TYPE_F1, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def value_iteration(V, delta, gamma, d_i0, d_i1, d_i2, d_f0, d_f1, P_xy):
    """W given policy."""
    J = d_i0['J']
    sizes_x = d_i1['sizes_i'][1:J + 1]
    sizes_s = d_i1['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i1['sizes'][J:J * 2]
    lab = d_f1['lab']
    mu = d_f1['mu']
    V_t = d_f0['tau'] * V  # Copies V
    time = d_f0['start_time']
    converged = False
    count = int(0)
    g = float(0)
    while ((not converged) and (count < d_f0['max_iter'])
           and (time < d_f0['max_time'])):
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s_valid'])):
                x, s = d_i2['x'][x_i], d_i2['s_valid'][s_i]
                state = np.sum(x * sizes_x_n + s * sizes_s_n)
                for i in nb.prange(J + 1):
                    state_i = (i * d_i1['sizes_i'][0]
                               + np.sum(x * sizes_x + s * sizes_s))
                    if x[i] == 0:
                        V_t[state] += lab[i] * (
                            get_w_i(V, i, x, s, state, gamma,
                                    d_i0, d_i1, d_f1, P_xy) - V[state])
                    else:
                        V_t[state] += gamma * (
                                get_w_i(V, i, x, s, state, gamma, d_i0,
                                        d_i1, d_f1, P_xy) - V[state])
                    if s[i] > 0:
                        next_state = state_i - sizes_s[i]
                        V_t[state] += (s[i] * mu[i]
                                       * (get_w_i(V, i, x, s, next_state, gamma,
                                                  d_i0, d_i1, d_f1, P_xy)
                                          - V[state]))
                V_t[state] /= d_f0['tau']
                delta[state] = abs(V_t[state] - V[state])
        if count % d_i0['convergence_check'] == 0:
            delta_max = np.float32(-np.inf)
            delta_min = np.float32(np.inf)
            for x_i in range(len(d_i2['x'])):
                for s_i in range(len(d_i2['s_valid'])):
                    x, s = d_i2['x'][x_i], d_i2['s_valid'][s_i]
                    state = np.sum(x * sizes_x_n + s * sizes_s_n)
                    if delta[state] > delta_max:
                        delta_max = delta[state]
                    if delta[state] < delta_min:
                        delta_min = delta[state]
                    if delta_max - delta_min > d_f0['e']:
                        break
                if delta_max - delta_min > d_f0['e']:
                    break
            converged = delta_max - delta_min < d_f0['e']
            g = (delta_max + delta_min) / 2 * d_f0['tau']
            # with objmode(time1='f8'):
            #     time = clock()
        if count % d_i0['print_modulo'] == 0:
            print(f'iter: {count}, delta: %.4f, d_min %.4f, d_max %.4f, '
                  f'g: %.4f' % delta_max - delta_min, g, delta_min, delta_max)
        count += 1
    return V_t, g, converged, count

# Value Iteration
name = 'Value Iteration'
V = np.zeros(env.size, dtype=np.float32)  # V_{t-1}
delta = np.ones(env.size, dtype=np.float32) * np.inf
count = 0
converged = False

start_VI = clock()
V, g, converged, count = value_iteration(V, delta, env.J, env.D, env.gamma,
                                         env.d_i1, env.d_i2, env.d_f0, env.d_f0,
                                         env.P_xy)
print(f"time_VI {start_VI - clock()}, g %.3f, converged {converged}, "
      f"count {count} " % g)
