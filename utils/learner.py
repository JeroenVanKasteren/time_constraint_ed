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
        self.stable = False
        if 'Pi' in kwargs:
            self.Pi = kwargs.get('Pi')

    @staticmethod
    def init_pi(env, method):
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
                states_ = states.copy()
                for x in range(1, env.D + 1):
                    states_[1 + i] = x  # x_i = x
                    for j in range(env.J):
                        if j != i:
                            if method == 'fcfs':
                                x_max = x
                            else:  # method == 'sdf':
                                x_max = min(env.D,
                                            max(0, env.gamma * env.t[j] -
                                                (env.gamma * env.t[i] - x)))
                            states_[1 + j] = slice(0, int(x_max + 1))
                    Pi[tuple(states_)] = i + 1
                states_ = np.append([0] * (1 + env.J), s)
                states_[0] = i
                Pi[tuple(states_)] = i + 1  # Admit arrival (of i)
            states = np.concatenate(([env.J], [0] * env.J, s), axis=0)
            Pi[tuple(states)] = env.NONE_WAITING  # x_i = 0 All i
        return Pi

    @staticmethod
    def init_w(env, V, W):
        """
        Write good description.
        """
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
                  f'delta: {delta_max - delta_min:.3f}, '
                  f'd_min: {delta_min:.3f}, d_max: {delta_max:.3f}, '
                  f'g: {g:.4f}')
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
    @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8,
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
    @nb.njit(nb.types.Tuple((nb.i4[:], nb.b1))(
        tp.f4[:], tp.f4[:], tp.i4[:], tp.i8, tp.i8, tp.f8, tp.i8,
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
        return Pi, stable == 0

    def policy_evaluation(self, env, V, W, Pi, g, name, iter=0):
        """Policy Evaluation."""
        inner_iter = 0
        stopped = False
        converged = False
        while not (stopped | converged):
            W = self.init_w(env, V, W)
            V = V.reshape(env.size)
            W = W.reshape(env.size_i)
            W = self.get_w(V, W, Pi, env.J, env.D, env.gamma,
                           env.d_i1, env.d_i2, env.d_f1, env.p_xy)
            V = V.reshape(env.dim)
            W = W.reshape(env.dim_i)
            V_t = self.get_v(env, V, W)
            if inner_iter % env.convergence_check == 0:
                converged, stopped, g = self.convergence(env, V_t, V, iter,
                                                         name, j=inner_iter)
            V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
            if inner_iter > env.max_iter:
                return V, g
            inner_iter += 1
        return V, g, converged, inner_iter

    def policy_iteration(s, env, g_mem=[], *kwargs):
        """Docstring."""
        if ('V' in kwargs) & ('Pi' in kwargs):
            s.V = kwargs.get('V')
            s.Pi = kwargs.get('Pi')
        else:
            s.V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
            s.Pi = s.init_pi(env, 'sdf')
        s.W = np.zeros(env.dim_i, dtype=np.float32)

        s.Pi = s.Pi.reshape(env.size_i)
        while not s.stable:
            s.V, s.g, _, _ = s.policy_evaluation(env, s.V, s.W, s.Pi, s.g,
                                                 'Policy Evaluation of PI',
                                                 s.iter)
            s.W = s.init_w(env, s.V, s.W)
            s.V = s.V.reshape(env.size)
            s.W = s.W.reshape(env.size_i)
            s.Pi, s.stable = s.policy_improvement(s.V, s.W, s.Pi, env.J, env.D,
                                                  env.gamma, env.KEEP_IDLE,
                                                  env.d_i1, env.d_i2, env.d_f1,
                                                  env.p_xy)
            s.V = s.V.reshape(env.dim)
            s.W = s.W.reshape(env.dim_i)
            if s.iter > env.max_iter:
                break
            s.iter += 1
            g_mem.append(s.g)
        s.Pi = s.Pi.reshape(env.dim_i)
        return g_mem


class ValueIteration:
    """Value Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F1 = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    def __init__(self, env, pi_learner):
        self.name = 'Value Iteration'
        self.pi_learner = pi_learner
        self.V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
        self.V_t = np.zeros(env.dim, dtype=np.float32)
        self.W = np.zeros(env.dim_i, dtype=np.float32)
        self.Pi = None
        self.g = 0
        self.iter = 0
        self.converged = False

    @staticmethod
    @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
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

    def value_iteration(s, env):
        stopped = False
        while not (stopped | s.converged):  # Update each state.
            s.W = s.pi_learner.init_w(env, s.V, s.W)
            s.V = s.V.reshape(env.size)
            s.W = s.W.reshape(env.size_i)
            s.W = s.get_w(s.V, s.W, env.J, env.D, env.gamma,
                          env.d_i1, env.d_i2, env.d_f1, env.p_xy)
            s.V = s.V.reshape(env.dim)
            s.W = s.W.reshape(env.dim_i)
            s.V_t = s.pi_learner.get_v(env, s.V, s.W)
            if s.iter % env.convergence_check == 0:
                s.converged, stopped, s.g = \
                    s.pi_learner.convergence(env, s.V_t, s.V, s.iter, s.name)
            s.V = s.V_t - s.V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
            s.iter += 1

    def get_policy(s, env):
        """Determine policy via Policy Improvement."""
        s.W = s.pi_learner.init_w(env, s.V, s.W)
        s.Pi = s.pi_learner.init_pi(env, 'sdf')
        s.V = s.V.reshape(env.size)
        s.W = s.W.reshape(env.size_i)
        s.Pi = s.Pi.reshape(env.size_i)
        s.Pi, _ = s.pi_learner.policy_improvement(s.V, s.W, s.Pi, env.J, env.D,
                                                  env.gamma, env.KEEP_IDLE,
                                                  env.d_i1, env.d_i2, env.d_f1,
                                                  env.p_xy)
        s.V = s.V.reshape(env.dim)
        s.W = s.W.reshape(env.dim_i)
        s.Pi = s.Pi.reshape(env.dim_i)


class OneStepPolicyImprovement:
    """One-step policy improvement."""

    def __init__(self, env, pi_learner=None):
        self.name = 'One-step Policy Improvement'
        self.V_app = self.get_v_app(env)
        if pi_learner is not None:
            self.pi_learner = pi_learner
            self.Pi = None
            self.V = np.zeros(env.dim, dtype=np.float32)
            self.g = 0
            self.iter = 0
            self.converged = False

    @staticmethod
    def get_v_app_i(env, i):
        """Calculate V for a single queue."""
        s, lab, mu, r = env.s_star[i], env.lab[i], env.mu[i], env.r[i]
        rho, a = env.rho[i], env.a[i]
        g = env.g[i]
        v_i = np.zeros(env.D + 1)

        # V(x) for x<=0, with V(-s)=0
        v_x_le_0 = lambda y: (1 - (y / a) ** s) / (1 - y / a) * np.exp(a - y)
        v_i[0] = (g - lab * r) / lab * quad_vec(v_x_le_0, a, np.inf)[0]
        # V(x) for x>0
        frac = (s * mu + env.gamma) / (lab + env.gamma)
        trm = np.exp(a) / a ** (s - 1) * gamma_fun(s) * reg_up_inc_gamma(s, a)
        x = np.arange(1, env.D + 1)
        v_i[x] = (v_i[0] + (s * mu * r - g) / (env.gamma * s * mu * (1-rho)**2)
                  * (lab + env.gamma - lab * x * (rho - 1)
                     - (lab + env.gamma) * frac ** x)
                  + 1 / (env.gamma * (rho - 1))
                  * (g - s*mu*r - env.gamma/lab * (g + (g - lab*r)/rho * trm))
                  * (-rho + frac ** (x - 1)))
        # -1_{x > gamma*t}[...]
        x = np.arange(env.gamma * env.t[i] + 1, env.D + 1).astype(int)
        v_i[x] -= env.c[i] / (env.gamma * (1 - rho) ** 2) * \
                  (lab + env.gamma - lab * (x - env.gamma * env.t[i] - 1)
                   * (rho - 1) - (lab + env.gamma) * frac ** (
                               x - env.gamma * env.t[i] - 1))
        return v_i

    def get_v_app(self, env):
        """Approximation of value function.
    
        Create a list V_memory with V_ij(x), i=class, j=#servers for all x.
        Note only j = s*_i, ..., s will be filled, rest zero
        """
        V_app = np.zeros(env.dim, dtype=np.float32)
        V = np.zeros((env.J, env.D + 1))
        for i in range(env.J):
            V[i, ] = self.get_v_app_i(env, i)
            for x in range(env.D + 1):
                states = [slice(None)] * (env.J * 2)
                states[i] = x
                V_app[tuple(states)] += V[i, x]
        return V_app

    def one_step_policy_improvement(s, env):
        """One Step of Policy Improvement."""
        W = np.zeros(env.dim_i, dtype=np.float32)
        W = s.pi_learner.init_w(env, s.V_app, W)
        s.Pi = s.pi_learner.init_pi(env, 'sdf')

        s.V_app = s.V_app.reshape(env.size)
        W = W.reshape(env.size_i)
        s.Pi = s.Pi.reshape(env.size_i)
        s.Pi, _ = s.pi_learner.policy_improvement(s.V_app, W, s.Pi, env.J,
                                                  env.D, env.gamma,
                                                  env.KEEP_IDLE, env.d_i1,
                                                  env.d_i2, env.d_f1, env.p_xy)
        s.V_app = s.V_app.reshape(env.dim)
        s.Pi = s.Pi.reshape(env.dim_i)

    def get_g(s, env, V):
        """Determine g via Policy Evaluation."""
        W = np.zeros(env.dim_i, dtype=np.float32)
        s.Pi = s.Pi.reshape(env.size_i)
        s.V, s.g, s.converged, s.iter = s.pi_learner.policy_evaluation(
            env, V, W, s.Pi, s.g, s.name)
        s.Pi = s.Pi.reshape(env.dim_i)
