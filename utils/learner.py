"""
Learner functions to take actions and learn from experience.

(The learner is also known as the agent or actor.)
The implemented learners are:
    - Policy Iteration (PI)
    - Value Iteration (VI)
    - One-Step Policy Iteration (OSPI)

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import numba as nb
import utils
from numba import types as tp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
from time import perf_counter as clock


class PolicyIteration:
    """Policy Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F1 = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    def __init__(self, **kwargs):
        self.name = 'Policy Iteration'
        self.g = kwargs.get('g', 0)
        self.iter = 0
        self.converged = False
        self.Pi = kwargs.get('Pi', None)
        self.V = None
        self.W = None


    @staticmethod
    def init_pi(env, method='sdf', order=None):
        """
        Take the longest waiting queue into service (or last queue if tied).
        Take arrivals into service when empty.
        """
        Pi = env.NOT_EVALUATED * np.ones(env.dim_i, dtype=np.int32)
        for s in env.s_states_v:
            states = np.append([slice(None)] * (1 + env.J), s)
            if np.sum(s) == env.S:
                Pi[tuple(states)] = env.SERVERS_FULL
                continue
            for i in range(env.J):
                for x in range(0, env.D + 1):
                    states_ = states.copy()
                    if x == 0:
                        states_[0] = i
                    states_[1 + i] = x  # x_i = x
                    for j in range(env.J):
                        if j != i:
                            if method == 'sdf':
                                x_max = min(env.D,
                                            max(0, env.gamma * env.t[j] -
                                                (env.gamma * env.t[i] - x)))
                            elif order is not None:
                                x_max = env.D if order[i] < order[j] else 0
                            else:  # FCFS
                                x_max = x
                            states_[1 + j] = slice(0, int(x_max + 1))
                    Pi[tuple(states_)] = i + 1
            states = np.concatenate(([env.J], [0] * env.J, s), axis=0)
            Pi[tuple(states)] = env.NONE_WAITING  # x_i = 0 All i
        return Pi

    @staticmethod
    def init_w(env, V):
        """
        Write good description.
        """
        W = np.zeros(env.dim_i, dtype=np.float64)
        for i in range(env.J):
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
            states = [slice(None)] * (1 + env.J * 2)
            for i in range(env.J):
                states[1 + i] = slice(int(env.gamma * env.t[i]) + 1, env.D + 1)
            for s in env.s_states:
                states[1 + env.J:] = s
                W[tuple(states)] -= env.P
        return W

    @staticmethod
    def convergence(env, V_t, V, i, name, j=-1):
        """Convergence check of valid states only."""
        delta_max = V_t[tuple([0] * (env.J * 2))] - V[tuple([0] * (env.J*2))]
        delta_min = delta_max.copy()
        for s in env.s_states_v:
            states = [slice(None)] * (env.J * 2)
            states[slice(env.J, env.J * 2)] = s
            diff = V_t[tuple(states)] - V[tuple(states)]
            delta_max = np.max([np.max(diff), delta_max])
            delta_min = np.min([np.min(diff), delta_min])
            if abs(delta_max - delta_min) > env.e:
                break
        converged = delta_max - delta_min < env.e
        max_iter = (i > env.max_iter) | (j > env.max_iter)
        max_time = (clock() - env.start_time) > env.max_time
        g = (delta_max + delta_min) / 2 * env.tau
        if (converged | (((i % env.print_modulo == 0) & (j == -1))
                         | (j % env.print_modulo == 0))):
            print(f'iter: {i}, inner_iter: {j}, '
                  f'delta: {delta_max - delta_min:.5f}, '
                  f'd_min: {delta_min:.5f}, d_max: {delta_max:.5f}, '
                  f'g: {g:.5f}')
            print(utils.tools.sec_to_time(clock() - env.start_time))
        if converged:
            iter = i if j == -1 else j
            print(f'{name} converged in {iter} iterations. '
                  f'g = {g:.4f}')
            print(utils.tools.sec_to_time(clock() - env.start_time))
        elif max_iter:
            print(f'{name} iter {i}, ({j}) reached max_iter '
                  f'({max_iter}), g ~ {g:.4f}')
            print(utils.tools.sec_to_time(clock() - env.start_time))
        elif max_time:
            print(f'{name} iter {i}, ({j}) reached max_time ({max_time}) '
                  f'g ~ %0.4f' % g)
            print(utils.tools.sec_to_time(clock() - env.start_time))
        return converged, max_iter | max_time, g

    @staticmethod
    @nb.njit(tp.f8[:](tp.f8[:], tp.f8[:], tp.i4[:], tp.i8, tp.i8, tp.f8,
                      DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
             parallel=True, error_model='numpy')
    def get_w(V, W, Pi, J, D, gamma,
              d_i1, d_i2, d_f1, p_xy):
        """W given policy."""
        sizes_x = d_i1['sizes_i'][1:J + 1]
        sizes_s = d_i1['sizes_i'][J + 1:J * 2 + 1]
        sizes_x_n = d_i1['sizes'][0:J]  # sizes Next state
        sizes_s_n = d_i1['sizes'][J:J * 2]
        r = d_f1['r']
        c = d_f1['c']
        t = d_f1['t']
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s'])):
                for i in nb.prange(J + 1):
                    x = d_i2['x'][x_i]
                    s = d_i2['s'][s_i]
                    state = (i * d_i1['sizes_i'][0] +
                             + np.sum(x * sizes_x + s * sizes_s))
                    if Pi[state] > 0:
                        j = Pi[state] - 1
                        W[state] = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
                        i_not_admitted = 0
                        if (i < J) and (i != j) and (x[i] < D):
                            i_not_admitted = sizes_x_n[i]
                        for y in range(x[j] + 1):
                            next_state = (np.sum(x * sizes_x_n + s * sizes_s_n)
                                          - (x[j] - y) * sizes_x_n[j]
                                          + i_not_admitted
                                          + sizes_s_n[j])
                            W[state] += p_xy[j, x[j], y] * V[next_state]
        return W

    @staticmethod
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

    @staticmethod
    @nb.njit(nb.types.Tuple((nb.i4[:], nb.b1, nb.i4))(
        tp.f8[:], tp.f8[:], tp.i4[:], tp.i8, tp.i8, tp.f8, tp.i8,
        DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
        parallel=True, error_model='numpy')
    def policy_improvement(V, W, Pi, J, D, gamma, keep_idle,
                           d_i, d_i2, d_f1, p_xy):
        """W given policy."""
        sizes_x = d_i['sizes_i'][1:J + 1]
        sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
        sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
        sizes_s_n = d_i['sizes'][J:J * 2]
        r = d_f1['r']
        c = d_f1['c']
        t = d_f1['t']
        stable = 0
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s'])):
               for i in nb.prange(J + 1):
                    x = d_i2['x'][x_i]
                    s = d_i2['s'][s_i]
                    state = i * d_i['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s)
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
                                next_state = (np.sum(
                                    x * sizes_x_n + s * sizes_s_n)
                                              - (x[j] - y) * sizes_x_n[j]
                                              + i_not_admitted
                                              + sizes_s_n[j])
                                value += p_xy[j, x[j], y] * V[next_state]
                            if value >= w:
                                Pi[state] = j + 1
                                w = value
                    if pi != Pi[state]:
                        stable = stable + 1  # binary operation allows reduction
        return Pi, stable == 0, stable

    def one_step_policy_improvement(s, env, V):
        """One Step of Policy Improvement."""
        s.W = s.init_w(env, V)
        s.Pi = s.init_pi(env)

        V = V.reshape(env.size)
        s.W = s.W.reshape(env.size_i)
        s.Pi = s.Pi.reshape(env.size_i)
        s.Pi, _, _ = s.policy_improvement(
            V, s.W, s.Pi, env.J, env.D, env.gamma,
            env.KEEP_IDLE, env.d_i1, env.d_i2, env.d_f1, env.p_xy)
        s.Pi = s.Pi.reshape(env.dim_i)
        s.W = s.W.reshape(env.dim_i)

    def policy_evaluation(self, env, V, g, name, n_iter=0):
        """Policy Evaluation."""
        assert self.Pi is not None, 'Policy not initialized.'
        assert self.Pi.shape == env.size_i, 'Policy shape mismatch.'
        inner_iter = 0
        stopped = False
        converged = False
        while not (stopped | converged):
            self.W = self.init_w(env, V)
            V = V.reshape(env.size)
            self.W = self.W.reshape(env.size_i)
            self.W = self.get_w(V, self.W, self.Pi, env.J, env.D, env.gamma,
                           env.d_i1, env.d_i2, env.d_f1, env.p_xy)
            V = V.reshape(env.dim)
            self.W = self.W.reshape(env.dim_i)
            V_t = self.get_v(env, V, self.W)
            if inner_iter % env.convergence_check == 0:
                converged, stopped, g = self.convergence(env, V_t, V, n_iter,
                                                         name, j=inner_iter)
            V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
            if inner_iter > env.max_iter:
                return V, g
            inner_iter += 1
        return V, g, converged, inner_iter

    def policy_iteration(s, env, g_mem=[], **kwargs):
        """Docstring."""
        s.V = kwargs.get('V', np.zeros(env.dim, dtype=np.float64))  # V_{t-1}
        s.Pi = kwargs.get('Pi', s.init_pi(env))
        max_pi_iter = kwargs.get('max_pi_iter', env.max_iter)
        stable = False

        s.Pi = s.Pi.reshape(env.size_i)
        while not stable:
            s.V, s.g, _, _ = s.policy_evaluation(env, s.V, s.g,
                                                 'Policy Evaluation of PI',
                                                 s.iter)
            s.W = s.init_w(env, s.V)
            s.V = s.V.reshape(env.size)
            s.W = s.W.reshape(env.size_i)
            s.Pi, stable, changes = s.policy_improvement(
                s.V, s.W, s.Pi, env.J, env.D, env.gamma, env.KEEP_IDLE,
                env.d_i1, env.d_i2, env.d_f1, env.p_xy)
            s.V = s.V.reshape(env.dim)
            s.W = s.W.reshape(env.dim_i)
            if s.iter > max_pi_iter:
                print(f'Policy Iteration reached max_iter ({max_pi_iter}),'
                      f' g ~ {s.g:.4f},  unstable changes ~ '
                      f'{changes}, {changes/env.size_i:.2f}%')
                break
            s.iter += 1
            g_mem.append(s.g)
        if stable:
            s.converged = True
        s.Pi = s.Pi.reshape(env.dim_i)
        return g_mem


class ValueIteration:
    """Value Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F1 = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    def __init__(self):
        self.name = 'Value Iteration'
        self.V = None
        self.g = 0
        self.iter = 0
        self.converged = False

    @staticmethod
    @nb.njit(tp.f8[:](tp.f8[:], tp.f8[:], tp.i8, tp.i8, tp.f8,
                      DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
             parallel=True, error_model='numpy')
    def get_w(V, W, J, D, gamma, d_i, d_i2, d_f1, p_xy):
        """W given policy."""
        sizes_x = d_i['sizes_i'][1:J + 1]
        sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
        sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
        sizes_s_n = d_i['sizes'][J:J * 2]
        r = d_f1['r']
        c = d_f1['c']
        t = d_f1['t']
        for x_i in nb.prange(len(d_i2['x'])):
            for s_i in nb.prange(len(d_i2['s'])):
                for i in nb.prange(J + 1):
                    x, s = d_i2['x'][x_i], d_i2['s'][s_i]
                    state = (i * d_i['sizes_i'][0]
                             + np.sum(x * sizes_x + s * sizes_s))
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
                                w += p_xy[j, x[j], y] * V[next_state]
                            if w > W[state]:
                                W[state] = w
        return W

    def value_iteration(s, env, pi_learner):
        if s.V is None:
            s.V = np.zeros(env.dim, dtype=np.float64)
        stopped = False
        while not (stopped | s.converged):  # Update each state.
            W = pi_learner.init_w(env, s.V)
            s.V = s.V.reshape(env.size)
            W = W.reshape(env.size_i)
            W = s.get_w(s.V, W, env.J, env.D, env.gamma,
                          env.d_i1, env.d_i2, env.d_f1, env.p_xy)
            s.V = s.V.reshape(env.dim)
            W = W.reshape(env.dim_i)
            V_t = pi_learner.get_v(env, s.V, W)
            if s.iter % env.convergence_check == 0:
                s.converged, stopped, s.g = \
                    pi_learner.convergence(env, V_t, s.V, s.iter, s.name)
            s.V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
            s.iter += 1


class OneStepPolicyImprovement:
    """One-step policy improvement."""

    def __init__(self):
        self.name = 'One-step Policy Improvement'
        self.V_app = None
        self.V = None
        self.g = 0
        self.iter = 0
        self.converged = False

    @staticmethod
    def get_v_app_i(env, i):
        """Calculate V for a single queue."""
        s, lab, mu, r = env.s_star[i], env.lab[i], env.mu[i], env.r[i]
        rho, a = env.rho[i], env.a[i]
        g = env.g[i]
        v_i = np.zeros(int(np.ceil(s)) + env.D + 1)

        # V(x) for x<=0, with V(-s)=0
        x = np.arange(-int(s) + 1, 0 + 1)  # V(x) for -s<x<=0
        x = np.append(x, s % 1) if s % 1 > 0 else x  # extra state if frac. s
        x_index = np.ceil(x).astype(int) + int(s)
        V_x_le_0 = lambda y: ((1 - (y / a) ** (x + s)) / (1 - y / a)
                              * np.exp(a - y))
        v_i[x_index] = (g - lab * r) / lab * quad_vec(V_x_le_0, a, np.inf)[0]

        # V(x) for x>0
        frac = (s * mu + env.gamma) / (lab + env.gamma)
        trm = np.exp(a) / a ** (s - 1) * gamma_fun(s) * reg_up_inc_gamma(s, a)
        x = np.arange(1, env.D + 1)
        x_index = x + int(np.ceil(s))
        v_i[x_index] = (v_i[int(np.ceil(s))]  # V(0)
                      + (s * mu * r - g) / (env.gamma * s * mu * (1-rho)**2)
                      * (lab + env.gamma - lab * x * (rho - 1)
                         - (lab + env.gamma) * frac ** x)
                      + 1 / (env.gamma * (rho - 1))
                      * (g - s * mu * r - env.gamma / lab
                         * (g + (g - lab * r) / rho * trm))
                      * (-rho + frac ** (x - 1)))
        # -1_{x > gamma*t}[...]
        x = np.arange(env.gamma * env.t[i] + 1, env.D + 1).astype(int)
        v_i[x + int(np.ceil(s))] -= (env.c[i] / (env.gamma * (1 - rho) ** 2)
                                     * (lab + env.gamma
                                        - lab * (x - env.gamma * env.t[i] - 1)
                                        * (rho - 1) - (lab + env.gamma) * frac
                                        ** (x - env.gamma * env.t[i] - 1)))
        return v_i

    def get_v_app_cons(self, env):
        """Approximation of value function.

        Conservative approach.
        Create a list V_memory with V_i(x), i=class, for all x.
        """
        v_app = np.zeros(env.dim, dtype=np.float64)
        for i in range(env.J):
            v_app_i = self.get_v_app_i(env, i)
            for x in range(env.D + 1):
                states = [slice(None)] * (env.J * 2)
                states[i] = x  # x_i = x
                v_app[tuple(states)] += v_app_i[x + int(np.ceil(env.s_star[i]))]
        return v_app

    def get_v_app(self, env):
        """Approximation of value function.
        Use that v_app_i(s) is known for s <= s_star.
        """
        v_app = np.zeros(env.dim, dtype=np.float64)
        for i in range(env.J):
            v_app_i = self.get_v_app_i(env, i)
            states = [slice(None)] * (env.J * 2)
            states[i] = 0  # x_i = 0
            for s in range(env.S + 1):
                states[env.J + i] = s  # s_i = s
                if s < env.s_star[i]:
                    v_app[tuple(states)] += v_app_i[s]
                else:  # s => s*_i
                    v_app[tuple(states)] += v_app_i[int(np.ceil(env.s_star[i]))]
            for x in range(1, env.D + 1):
                states = [slice(None)] * (env.J * 2)
                states[i] = x  # x_i = x
                v_app[tuple(states)] += v_app_i[x + int(np.ceil(env.s_star[i]))]
        return v_app

    @staticmethod
    def calc_dx(env, i, x, s, v_app_i):
        """Calculate dx for v_app."""
        s_ceil = np.ceil(env.s_star).astype(int)
        if (s < env.s_star[i]) or (x == env.D):
            return ((v_app_i[x + s_ceil[i]]
                     - v_app_i[x + s_ceil[i] - 1]) / (env.S + 1))
        else:
            return ((v_app_i[x + s_ceil[i] + 1]
                   - v_app_i[x + s_ceil[i]])
                  / (env.S + 1))

    def get_v_app_lin(self, env, method='abs'):
        """Approximation of value function.
        Use that v_app_i(s) is known for s <= s_star.
        And interpolate for s > s_star.
        """
        s_ceil = np.ceil(env.s_star).astype(int)
        s_int = env.s_star.astype(int)
        v_app = np.zeros(env.dim, dtype=np.float64)
        for i in range(env.J):
            v_app_i = self.get_v_app_i(env, i)
            for x in range(env.D + 1):
                for s in range(env.S + 1):
                    states = [slice(None)] * (env.J * 2)
                    states[i] = x  # x_i = x
                    states[env.J + i] = s  # s_i = s
                    if s < env.s_star[i]:
                        if x == 0:
                            v_app[tuple(states)] += v_app_i[s]
                        else:  # x > 0
                            if method == 'abs':
                                v_app[tuple(states)] += (
                                    v_app_i[s] + (v_app_i[x + s_ceil[i]]
                                                  - v_app_i[s_ceil[i]]))
                            else:
                                dx = self.calc_dx(env, i, x, s, v_app_i)
                                v_app[tuple(states)] += (
                                        v_app_i[x + s_ceil[i] - 1]
                                        + dx * (s + env.S + 1 - s_int[i]))
                    else:  # s*_i <= s <= S
                        dx = self.calc_dx(env, i, x, s, v_app_i)
                        v_app[tuple(states)] += (v_app_i[x + s_ceil[i]]
                                                 + dx * (s - s_int[i]))
        return v_app

    def calc_v_app_dp(self, env):
        """Approximation of value function with an approximate OSPI in dynamic
        programming fashion.
        """
        s_ceil = np.ceil(env.s_star).astype(int)
        v_dp = np.zeros((env.J, env.D + 1, env.S + 1))
        for i in range(env.J):
            # init v_dp matrix
            v_app_i = self.get_v_app_i(env, i)
            f = env.s_star[i] % 1
            for s in range(1, s_ceil[i] + 1):
                if f == 0:
                    v_dp[i, 0, s] = v_app_i[s]
                else:
                    v_dp[i, 0, s] = f * v_app_i[s] + (1 - f) * v_app_i[s - 1]
            for x in range(1, env.D + 1):
                v_dp[i, x, s_ceil[i]] = v_app_i[x + s_ceil[i]]
                for s in range(s_ceil[i] - 1, -1, -1):  # s = {0, ..., s_ceil}
                    r_xs = env.r[i]
                    if x > env.gamma * env.t[i]:
                        r_xs -= env.c[i]
                    r_xs = f * r_xs if s == 0 else r_xs
                    v_dp[i, x, s] = (r_xs + sum(env.p_xy[i, x, :x+1]
                                                * v_dp[i, :x + 1, s + 1])
                                     - env.g[i] / env.tau)
        return v_dp

    @staticmethod
    def calc_h(env, i, s_v):
        h = env.s_star[i] - (env.S - sum(s_v)) * env.s_star[i] / env.S
        # round floating point errors, ensure h>=0 if -1e16 negatives
        h = max(0, np.floor(h * 1e10) / 1e10)
        n = int(env.s_star[i] - h)
        return h, n

    @staticmethod
    def calc_f(env, i, x, h, n):
        s_frac = (env.s_star[i] % 1)
        if (x > 0) and (h >= s_frac):
            return h - (env.s_star[i] - (n + 1))
        elif ((x == 0) and (h <= int(env.s_star[i]))) or s_frac == 0:
            return h % 1
        else:  # (x = 0 and int(s_star) < h) or (x > 0 and h < 1)
            return (h % 1) / s_frac

    def get_v_app_dp(self, env):
        """Approximation of value function with an approximate OSPI in dynamic
        programming fashion.
        """
        v_app = np.zeros(env.dim, dtype=np.float64)
        v_dp = self.calc_v_app_dp(env)
        for s_v in env.s_states_v:  # for every combination of s
            for i in range(env.J):
                states = [slice(None)] * (env.J * 2)
                states[env.J:] = s_v
                h, n = self.calc_h(env, i, s_v)
                for x in range(env.D + 1):
                    states[i] = x  # x_i = x
                    f = self.calc_f(env, i, x, h, n)
                    assert 0 <= f <= 1
                    s_i = int(env.s_star[i])
                    v_app[tuple(states)] += (f * v_dp[i, x, s_i - n] + (1 - f)
                                             * v_dp[i, x, s_i - n - 1])
        return v_app

    def get_g(s, env, V, pi_learner):
        """Determine g via Policy Evaluation."""
        pi_learner.Pi = pi_learner.Pi.reshape(env.size_i)
        s.V, s.g, s.converged, s.iter = pi_learner.policy_evaluation(
            env, V, s.g, s.name)
        pi_learner.Pi = pi_learner.Pi.reshape(env.dim_i)
