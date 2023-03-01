import numpy as np
import numba as nb
from numba import types as tp
from numpy import array, arange, zeros
from OtherTests.init import Env
import timeit

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(0)
env = Env(J=2, S=4, load=0.5, gamma=10., D=25, P=1000, e=1e-4, trace=True,
          print_modulo=100)
# env = Env(J=1, S=4, mu=array([1.5]), lmbda=array([4]), t=array([2]), P=0,
#           gamma=2, D=30, e=1e-5, trace=False)

DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector


def init_w(env, V, W):
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
        states = [slice(None)]*(1 + env.J*2)
        for i in arange(env.J):
            states[1 + i] = slice(int(env.gamma * env.t[i]) + 1, env.D+1)
        for s in env.s_states:
            states[1 + env.J:] = s
            W[tuple(states)] -= env.P
    return W


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
                    for y in arange(x[j] + 1):
                        next_state = (np.sum(x*sizes_x_n + s*sizes_s_n)
                                      - (x[j] - y) * sizes_x_n[j]
                                      + i_not_admitted
                                      + sizes_s_n[j])
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def get_w_old(V, W, Pi, J, D, gamma,
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
                    next_x = x.copy()  # copy part of old code
                    if (i < J) and (i != j) and (x[i] < D):
                        next_x[i] = next_x[i] + 1
                    for y in arange(x[j] + 1):
                        next_x[j] = y
                        next_s = s.copy()
                        next_s[j] += 1
                        next_state = np.sum(next_x*sizes_x_n + next_s*sizes_s_n)
                        W[state] += P_xy[j, x[j], y] * V[next_state]
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

# Same output?
W_rand = array(np.random.rand(env.size_i), dtype=np.float32)
V = array(np.random.rand(env.size), dtype=np.float32)
Pi = env.init_pi()
Pi = Pi.reshape(env.size_i)
W = W_rand.copy()
W_new = get_w(V, W, Pi, env.J, env.D, env.gamma, d_i1, d_i2, d_f, env.P_xy)
W = W_rand.copy()
W_old = get_w_old(V, W, Pi, env.J, env.D, env.gamma, d_i1, d_i2, d_f, env.P_xy)
print(np.allclose(W_new, W_old))  # True

name = "Test W"
V = zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = zeros(env.dim_i, dtype=np.float32)
Pi = env.init_pi()
Pi = Pi.reshape(env.size_i)

count = 0
n = 10
env.timer(True, name, env.trace)
for test_range in range(n):  # Update each state.
    W = init_w(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    W = get_w(V, W, Pi, env.J, env.D, env.gamma, d_i1, d_i2, d_f, env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
print(env.timer(False, name, env.trace)/n)

V = zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = zeros(env.dim_i, dtype=np.float32)
W = init_w(env, V, W)
Pi = env.init_pi()
V = V.reshape(env.size)
W = W.reshape(env.size_i)
Pi = Pi.reshape(env.size_i)
print(np.mean(timeit.repeat("get_w(V, W, Pi, env.J, env.D, env.gamma, d_i1, "
                            "d_i2, d_f, env.P_xy)",
                            "from __main__ import get_w,"
                            "DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F,"
                            "V, W, Pi, env, d_i1, d_i2, d_f;"
                            "import numpy as np; import numba as nb",
                            repeat=5, number=3))/3)


# J=2, S=4, load=0.5, gamma=10., D=25, P=1000
# Timing get_w(...), which is almost all time per iteration
# Old code (copying next state)
# Python: 7.84, Numba: 0.13
# Without copying
# Python: 6.7, Numba: 0.072
