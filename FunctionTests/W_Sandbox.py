
import numpy as np
import numba as nb
from numba import types as tp
from numpy import array, arange, zeros
from OtherTests.init import Env

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(0)
env = Env(J=1, S=2, load=0.5, gamma=2., D=5, P=1000, e=1e-4, trace=True,
          print_modulo=100)
# env = Env(J=1, S=4, mu=array([1.5]), lmbda=array([4]), t=array([2]), P=0,
#           gamma=2, D=30, e=1e-5, trace=False)

DICT_TYPE_I = tp.DictType(tp.unicode_type, tp.i8[:])  # int vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float vector

def init_W(env, V, W):
    for i in arange(env.J):
        states = np.append(i, [slice(None)] * (env.J * 2))
        states[1 + i] = slice(env.D)
        next_states = [slice(None)] * (env.J * 2)
        next_states[i] = slice(1, env.D + 1)
        W[tuple(states)] = V[tuple(next_states)]
        states[1 + i] = env.D
        next_states[i] = env.D
        W[tuple(states)] = V[tuple(next_states)]
    W[env.J] = V
    if env.P > 0:
        states = [[slice(None)]*(1 + env.J*2)]
        for i in arange(env.J):
            states[1 + i] = slice(int(env.gamma * env.t[i]) + 1, env.D+1)
        for s in env.s_states:
            states[1 + env.J:] = s
            W[tuple(states)] -= env.P
    return W


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I, DICT_TYPE_F, tp.f4[:, :, :]))
def get_W(V, W, Pi, J, D, gamma,
          d_f, d_i, P_xy):
    """W given policy."""
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i['sizes'][0:J]
    sizes_s_n = d_i['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    for s_i in nb.prange(len(d_i['s'])):
        for x_i in nb.prange(len(d_i['x'])):
            for i in np.arange(J):
                x = d_i['x'][x_i]
                s = d_i['s'][s_i]
                state = i * d_i['sizes_i'][0] + np.sum(x*sizes_x + s*sizes_s)
                if Pi[state] > 0:
                    j = Pi[state] - 1
                    W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                    next_x = x.copy()
                    for y in arange(x[j] + 1):
                        next_x[j] = y
                        if (i < J) and (i != j):
                            next_x[i] = np.min(next_x[i] + 1, D)
                        next_s = s.copy()
                        next_s[j] += 1
                        next_state = np.sum(next_x*sizes_x_n+next_s*sizes_s_n)
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W

d_i1 = nb.typed.Dict.empty(key_type=tp.unicode_type, value_type=tp.i8[:])
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
env.timer(True, name, env.trace)
V = zeros(env.dim)  # V_{t-1}
W = zeros(env.dim_i)
Pi = env.init_Pi()

count = 0
env.timer(True, name, env.trace)
for test_range in range(10):  # Update each state.
    W = init_W(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    Pi = Pi.reshape(env.size_i)
    W = get_W(V, W, Pi)
    W = W.reshape(env.dim_i)
env.timer(False, name, env.trace)
