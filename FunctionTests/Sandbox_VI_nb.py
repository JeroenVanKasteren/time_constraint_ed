"""
Sandbox Value Iteration
"""

import numpy as np
import numba as nb
from numba import types as tp
from FunctionTests.Sandbox_PI import init_pi, policy_improvement
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration
from Insights import plot_pi, plot_v

np.set_printoptions(precision=4, linewidth=150, suppress=True)

env = Env(J=1, gamma=30, P=1e3, e=1e-5, seed=seed,
          max_time='00:01:30', convergence_check=10, print_modulo=100)
# env = Env(J=1, S=1, mu=array([3]), lab=array([1]), t=array([1]), P=1e3,
#           gamma=1, D=5, e=1e-4, trace=True, print_modulo=100,
#           max_iter=5)
pi_learner = PolicyIteration()

DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

def get_w_i(i, x, s, state_i):
    w_res = V[state]
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
                w_res = w  # TODO, does this not copy it?
    return W

@staticmethod
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
                state = np.sum(x * sizes_x_n + s * sizes_s_n)
                state_i = i * d_i1['sizes_i'][0] + np.sum(
                    x * sizes_x + s * sizes_s)
                W[state_i] = V[state]
                if P > 0:
                    count = 0
                    for k in range(J):
                        count += x[i] > t[i]*gamma
                    if count == J:  # by definition, sum(s_i) = S
                        W[state_i] -= P
                for j in range(J):
                    if (x[j] > 0) or (j == i):
                        w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        i_not_admitted = 0
                        if (i < J) and (i != j) and (x[i] < D):
                            i_not_admitted = sizes_x_n[i]
                        for y in range(x[j] + 1):
                            next_state = (np.sum(
                                x * sizes_x_n + s * sizes_s_n)
                                          - (x[j] - y) * sizes_x_n[j]
                                          + i_not_admitted
                                          + sizes_s_n[j])
                            w += P_xy[j, x[j], y] * V[next_state]
                        if w > W[state_i]:
                            W[state_i] = w
    return W


@staticmethod
@nb.njit
def convergence(env, V_t, V, i, name, j=-1):
    """Convergence check of valid states only."""
    delta_max = V_t[tuple([0] * (env.J * 2))] - V[tuple([0] * (env.J*2))]
    delta_min = delta_max.copy()
    for x_i in nb.prange(len(d_i2['x'])):
        for s_i in nb.prange(len(d_i2['s'])):
            for i in nb.prange(J + 1):
                x = d_i2['x'][x_i]
                s = d_i2['s'][s_i]
                diff = V_t[tuple(states)] - V[tuple(states)]
                delta_max = np.max([np.max(diff), delta_max])
                delta_min = np.min([np.min(diff), delta_min])
                if abs(delta_max - delta_min) > env.e:
                    break  # TODO, how to break all loops?
    converged = delta_max - delta_min < env.e
    max_iter = (i > env.max_iter) | (j > env.max_iter)
    max_time = (clock() - env.start_time) > env.max_time
    g = (delta_max + delta_min) / 2 * env.tau
    if (converged | (((i % env.print_modulo == 0) & (j == -1))
                     | (j % env.print_modulo == 0))):
        print("iter: ", i,
              "inner_iter: ", j,
              ", delta: ", round(delta_max - delta_min, 2),
              ', D_min', round(delta_min, 2),
              ', D_max', round(delta_max, 2),
              ", g: ", round(g, 4))
    if converged:
        if j == -1:
            print(name, 'converged in', i, 'iterations. g=', round(g, 4))
        else:
            print(name, 'converged in', j, 'iterations. g=', round(g, 4))
    elif max_iter:
        print(name, 'iter:', i, '(', j, ') reached max_iter =',
              max_iter, ', g~', round(g, 4))
    elif max_time:
        print(name, 'iter:', i, '(', j, ') reached max_time =',
              max_time, ', g~', round(g, 4))
    return converged, max_iter | max_time, g


@nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
                  DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
         parallel=True, error_model='numpy')
def value_iteration(V, W, J, D, gamma, d_i1, d_i2, d_f, P_xy):
    """W given policy."""
    sizes_x = d_i1['sizes_i'][1:J + 1]
    sizes_s = d_i1['sizes_i'][J + 1:J * 2 + 1]
    sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
    sizes_s_n = d_i1['sizes'][J:J * 2]
    lab = d_f['lab']
    mu = d_f['mu']
    r = d_f['r']
    c = d_f['c']
    t = d_f['t']
    V_t = env.tau * V  # TODO, does this also copy it in numba?
    while not converged:  # Update each state.
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s'])):
                for i in nb.prange(J + 1):
                    x, s = d_i2['x'][x_i], d_i2['s'][s_i]
                    state = np.sum(x * sizes_x_n + s * sizes_s_n)
                    state_i = i * d_i1['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s)
                    if x[i] == 0:  # TODO, no labda and mu
                        V_t[state] += lab[i] * (get_w_i(i, x, s, state_i)
                                                - V[state])
                    else:
                        V_t[state] += gamma * (get_w_i(i, x, s, state_i)
                                               - V[state])
                    if s[i] > 0:
                        next_state = state_i - sizes_s[i]
                        V_t[state] += s[i] * mu[i] * \
                                      (get_w_i(i, x, s, next_state) - V[state])
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s'])):  # TODO, sum s == S
                for i in nb.prange(J + 1):
                    x, s = d_i2['x'][x_i], d_i2['s'][s_i]
                    state = np.sum(x * sizes_x_n + s * sizes_s_n)
                    next_state = state + sizes_x[i]
                    state_i = i * d_i1['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s)
                    if x[i] == 0:  # TODO, no labda and mu
                        V_t[state] += lab[i] * (V[next_state] - V[state])
                    else:
                        V_t[state] += gamma * (V[next_state] - V[state])
                    state_i = i * d_i1['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s) - sizes_s[i]
                    V_t[state] += s[i] * mu[i] * (
                                get_w(i, x, s, state_i) - V[state])
        V_t = V_t / env.tau
        if s.count % env.convergence_check == 0:
            s.converged, stopped, s.g = \
                s.pi_learner.convergence(env, s.V_t, s.V, s.count, s.name)
        s.V = s.V_t - s.V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
        if s.count > env.max_iter:
            break
        s.count += 1


# Value Iteration
name = 'Value Iteration'
V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
W = np.zeros(env.dim_i, dtype=np.float32)
Pi = init_pi(env)

count = 0
converged = False

env.timer(True, name, env.trace)
value_iteration(V, W, env.J, env.D, env.gamma, env.d_i1, env.d_i2, env.d_f,
                env.P_xy)
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
