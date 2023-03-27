import numpy as np
import numba as nb
from numba import types as tp
from OtherTests.init import Env
import timeit

np.set_printoptions(precision=4, linewidth=150, suppress=True)

N = 10
repeats = 5
n = 1
np.random.seed(42)
env = Env(J=3, S=2, load=0.5, gamma=10., D=10, P=1000, e=1e-4, trace=True,
          print_modulo=100)
# env = Env(J=1, S=4, mu=array([1.5]), lmbda=array([4]), t=array([2]), P=0,
#           gamma=2, D=30, e=1e-5, trace=False)

DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w(V, W, J, D, gamma, d_i, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    for x_i in nb.prange(len(d_i2['x'])):
        for s_i in nb.prange(len(d_i2['s'])):
            for i in nb.prange(J + 1):
                x = d_i2['x'][x_i]
                s = d_i2['s'][s_i]
                state = i * d_i['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
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

name = "Test W"
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)

env.timer(True, name, env.trace)
for test_range in range(n):  # Update each state.
    W = env.init_w(V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    W = get_w(V, W, env.J, env.D, env.gamma, d_i1, d_i2, d_f, env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
print(env.timer(False, name, env.trace)/n)

V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
W = env.init_w(V, W)
V = V.reshape(env.size)
W = W.reshape(env.size_i)

imports_and_vars = globals()
imports_and_vars.update(locals())
print(np.mean(timeit.repeat("get_w(V, W, env.J, env.D, env.gamma, d_i1, "
                            "d_i2, d_f, env.P_xy)",
                            globals=imports_and_vars,
                            repeat=repeats, number=N))/N)

# J=2, S=4, load=0.5, gamma=10., D=25, P=1000
# Timing get_w(...), which is almost all time per iteration
# 0.128 seconds per iteration
#
# """
# Next try global variables
# Numba considers global variables as compile-time constants.
# Otherwise your variable should be a function argument.
# """
#
# env = Env(J=3, S=2, load=0.5, gamma=10., D=10, P=1000, e=1e-4, trace=True,
#           print_modulo=100)
# # env = Env(J=1, S=4, mu=array([1.5]), lmbda=array([4]), t=array([2]), P=0,
# #           gamma=2, D=30, e=1e-5, trace=False)
#
#
# s_states = env.s_states
# s_n = len(s_states)
# x_states = env.x_states
# x_n = len(x_states)
# t = env.t
# c = env.c
# r = env.r
# J = env.J
# D = env.D
# gamma = env.gamma
# P_xy = env.P_xy
#
# sizes_i_0 = env.sizes_i[0]
# sizes_x = env.sizes_i[1:J + 1]
# sizes_s = env.sizes_i[J + 1:J * 2 + 1]
# sizes_x_n = env.sizes[0:J]  # sizes Next state
# sizes_s_n = env.sizes[J:J * 2]
#
#
# @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:]),
#          parallel=True, error_model='numpy')
# def get_w(V, W):
#     """W given policy."""
#     x_i, s_i, i, j, i_not_admitted = 0, 0, 0, 0, 0
#     state, next_state, w = 0.0, 0.0, 0.0
#     s, x = [0]*J, [0]*J
#     for x_i in nb.prange(x_n):
#         for s_i in nb.prange(s_n):
#             for i in nb.prange(J + 1):
#                 s = s_states[s_i]
#                 x = x_states[x_i]
#                 state = i * sizes_i_0 + np.sum(x*sizes_x + s*sizes_s)
#                 for j in range(J):
#                     if (x[j] > 0) or (j == i):
#                         w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
#                         i_not_admitted = 0
#                         if (i < J) and (i != j) and (x[i] < D):
#                             i_not_admitted = sizes_x_n[i]
#                         for y in range(x[j] + 1):
#                             next_state = (np.sum(x*sizes_x_n + s*sizes_s_n)
#                                           - (x[j] - y) * sizes_x_n[j]
#                                           + i_not_admitted
#                                           + sizes_s_n[j])
#                             w += P_xy[j, x[j], y] * V[next_state]
#                         if w > W[state]:
#                             W[state] = w
#     return W
#
#
# name = "Test W"
# V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
# W = np.empty(env.dim_i, dtype=np.float32)
#
# env.timer(True, name, env.trace)
# for test_range in range(n):  # Update each state.
#     W = env.init_w(V, W)
#     V = V.reshape(env.size)
#     W = W.reshape(env.size_i)
#     W = get_w(V, W)
#     V = V.reshape(env.dim)
#     W = W.reshape(env.dim_i)
# print(env.timer(False, name, env.trace)/n)
#
# V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
# W = np.empty(env.dim_i, dtype=np.float32)
# W = env.init_w(V, W)
# V = V.reshape(env.size)
# W = W.reshape(env.size_i)
#
# imports_and_vars = globals()
# imports_and_vars.update(locals())
# print(np.mean(timeit.repeat("get_w(V, W)",
#                             globals=imports_and_vars,
#                             repeat=repeats, number=N))/N)
#
# # J=2, S=4, load=0.5, gamma=10., D=25, P=1000
# # Timing get_w(...), which is almost all time per iteration
# # 0.1258 seconds per iteration
