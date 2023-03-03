import numpy as np
import numba as nb
from numba import types as tp
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


@nb.njit(nb.types.Tuple((nb.i4[:], nb.b1))(
    tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8, tp.i8,
    DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
    parallel=True, error_model='numpy')
def get_w(V, W, Pi, J, D, gamma, keep_idle,
          d_i, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i['sizes_i'][1:J + 1]
    sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i['sizes'][J:J * 2]
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    stable = True
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
                    stable = False
    return Pi, stable


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
Pi = env.init_pi()
Pi = Pi.reshape(env.size_i)

n = 10
env.timer(True, name, env.trace)
for test_range in range(n):  # Update each state.
    W = env.init_w(V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    Pi, stable = get_w(V, W, Pi, env.J, env.D, env.gamma, env.KEEP_IDLE,
                       d_i1, d_i2, d_f, env.P_xy)
    V = V.reshape(env.dim)
    W = W.reshape(env.dim_i)
print(env.timer(False, name, env.trace)/n)

V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
W = env.init_w(V, W)
Pi = env.init_pi()
V = V.reshape(env.size)
W = W.reshape(env.size_i)
Pi = Pi.reshape(env.size_i)
print(np.mean(timeit.repeat("get_w(V, W, Pi, env.J, env.D, env.gamma, "
                            "env.KEEP_IDLE, d_i1, d_i2, d_f, env.P_xy)",
                            "from __main__ import get_w,"
                            "DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F,"
                            "V, W, Pi, env, d_i1, d_i2, d_f;"
                            "import numpy as np; import numba as nb",
                            repeat=5, number=3))/3)

# J=2, S=4, load=0.5, gamma=10., D=25, P=1000
# Timing get_w(...), which is almost all time per iteration
# 0.12
