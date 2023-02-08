
import numpy as np
import numba as nb
from numpy import array, arange, zeros
from OtherTests.init import Env

np.set_printoptions(precision=4, linewidth=150, suppress=True)

np.random.seed(0)
env = Env(J=1, S=2, Rho=0.5, gamma=2, D=5, P=1000, e=1e-4, trace=True,
          print_modulo=100)
# env = Env(J=1, S=4, mu=array([1.5]), lmbda=array([4]), t=array([2]), P=0,
#           gamma=2, D=30, e=1e-5, trace=False)


Not_Evaluated = env.NOT_EVALUATED
Servers_Full = env.SERVERS_FULL
None_Waiting = env.NONE_WAITING
Keep_Idle = env.KEEP_IDLE

def init_W(env, V, W):
    for i in arange(env.J):
        states = np.append(i, [slice(None)] * (env.J * 2))
        states[1 + i] = slice(env.D)
        next_states = [slice(None)] * (env.J * 2)
        next_states[i] = slice(1, env.D + 1)
        W[tuple(states)] = V[tuple(next_states)]
        states[1 + i] = env.D
        next_states[i] = env.D
        W[tuple(states)] = V[tuple(next_states)] - env.P
    W[env.J] = V
    if env.P > 0:
        states = [[slice(None)]*(1 + env.J*2)]
        for i in arange(env.J):
            states[i] = slice(int(env.gamma * env.t[i]), env.D+1)
        for s in env.s_states:
            states[1 + env.J:] = s
            W[tuple(states)] -= env.P
    return W


@nb.njit nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:],
                    nb.int, nb.int, nb.float32,
                    nb.float32[:], nb.float32[:], nb.float32[:],
                    nb.int[:], nb.int[:], nb.int[:], nb.int[:], nb.int[:],
                    nb.float32[:])
def get_W(V, W, Pi,
        J, D, gamma,
        t, c, r,
        sizes, sizes_i, dim_i, s_states, x_states,
        P_xy):
    """W given policy."""
    for s in s_states:
        for x in x_states:
            for i in arange(J+1):
                state = i * sizes_i[0] + np.sum(x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                if Pi[state] > 0:
                    j = Pi[state] - 1
                    W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                    next_x = x.copy()
                    for y in arange(x[j] + 1):
                        next_x[j] = y
                        if (i < J) and (i != j):
                            next_x[i] = min(next_x[i] + 1, D)
                        next_s = s.copy()
                        next_s[j] += 1
                        next_state = np.sum(next_x * sizes[0:J] + next_s * sizes[J:J * 2])
                        W[state] += P_xy[j, x[j], y] * V[next_state]
    return W.reshape(dim_i)


name = "Test W"
env.timer(True, name, env.trace)
V = zeros(env.dim)  # V_{t-1}
W = zeros(env.dim_i)
Pi = env.init_Pi()

count = 0
env.timer(True, name, env.trace)
for i in range(10):  # Update each state.
    W = init_W(env, V, W)
    V = V.reshape(env.size)
    W = W.reshape(env.size_i)
    Pi = Pi.reshape(env.size_i)
    W = get_W(V, W, Pi)
    W = W.reshape(env.dim_i)
env.timer(False, name, env.trace)
