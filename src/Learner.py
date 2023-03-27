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
from numba import types as tp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec


class PolicyIteration():
    """Policy Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    x_states, s_states, P_xy, keep_idle, sizes_i_0, sizes_x, sizes_s, \
        sizes_x_n, sizes_s_n, t, c, r, J, D, x_n, s_n, gamma = tuple([None]*17)

    def __init__(self, env):
        self.name = 'Policy Iteration'
        self.g = 0
        self.count = 0
        self.stable = False

        self.s_states = env.s_states
        self.s_n = len(self.s_states)
        self.x_states = env.x_states
        self.x_n = len(self.x_states)
        self.t = env.t
        self.c = env.c
        self.r = env.r
        self.J = env.J
        self.D = env.D
        self.gamma = env.gamma
        self.P_xy = env.P_xy
        self.keep_idle = env.KEEP_IDLE

        self.sizes_i_0 = env.sizes_i[0]
        self.sizes_x = env.sizes_i[1:J + 1]
        self.sizes_s = env.sizes_i[J + 1:J * 2 + 1]
        self.sizes_x_n = env.sizes[0:J]  # sizes Next state
        self.sizes_s_n = env.sizes[J:J * 2]

    @staticmethod
    def init_pi(env):
        """
        Take the longest waiting queue into service (or last queue if tied).
        Take arrivals directly into service.
        """
        Pi = env.NOT_EVALUATED * np.ones(env.dim_i, dtype=int)
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
                            states_[1 + j] = slice(0, x + 1)  # 0 <= x_j <= x_i
                    Pi[tuple(states_)] = i + 1
                states_ = states.copy()
                states_[0] = i
                states_[1 + i] = 0
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
        g = (delta_max + delta_min) / 2 * env.tau
        if ((converged & env.trace)
                | (env.trace & (((i % env.print_modulo == 0) & (j == -1))
                                 | (j % env.print_modulo == 0)))):
            print("iter: ", i,
                  "inner_iter: ", j,
                  ", delta: ", round(delta_max - delta_min, 2),
                  ', D_min', round(delta_min, 2),
                  ', D_max', round(delta_max, 2),
                  ", g: ", round(g, 4))
        elif converged:
            if j == -1:
                print(name, 'converged in', i, 'iterations. g=', round(g, 4))
            else:
                print(name, 'converged in', j, 'iterations. g=', round(g, 4))
        elif max_iter:
            print(name, 'iter:', i, 'reached max_iter =', max_iter, ', g~',
                  round(g, 4))
        return converged | max_iter, g

    @staticmethod
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
                    state = i * d_i['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s)
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
                            W[state] += P_xy[j, x[j], y] * V[next_state]
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
        DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F, tp.f8[:, :, :]),
        parallel=True, error_model='numpy')
    def policy_improvement(V, W, Pi, J, D, gamma, keep_idle,
                           d_i, d_i2, d_f, P_xy):
        """W given policy."""
        sizes_x = d_i['sizes_i'][1:J + 1]
        sizes_s = d_i['sizes_i'][J + 1:J * 2 + 1]
        sizes_x_n = d_i['sizes'][0:J]  # sizes Next state
        sizes_s_n = d_i['sizes'][J:J * 2]
        r = d_f['r']
        c = d_f['c']
        t = d_f['t']
        stable = 0
        for s_i in nb.prange(len(d_i2['s'])):
           for x_i in nb.prange(len(d_i2['x'])):
               for i in nb.prange(J + 1):
                    # for s_i in range(len(d_i2['s'])):
                    #     for x_i in range(len(d_i2['x'])):
                    #         for i in range(J + 1):
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
                                value += P_xy[j, x[j], y] * V[next_state]
                            if value > w:
                                Pi[state] = j + 1
                                w = value
                    if pi != Pi[state]:
                        stable = stable + 1  # binary operation allows reduction
        return Pi, stable == 0

    def policy_evaluation(self, env, V, W, Pi, g, name, count=0):
        """Policy Evaluation."""
        inner_count = 0
        converged = False
        while not converged:
            W = self.init_w(env, V, W)
            V = V.reshape(env.size)
            W = W.reshape(env.size_i)
            W = self.get_w(V, W, Pi, env.J, env.D, env.gamma,
                           env.d_i1, env.d_i2, env.d_f, env.P_xy)
            V = V.reshape(env.dim)
            W = W.reshape(env.dim_i)
            V_t = self.get_v(env, V, W)
            if inner_count % env.convergence_check == 0:
                converged, g = self.convergence(env, V_t, V, count, name,
                                                j=inner_count)
            V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
            if count > env.max_iter:
                return V, g
            inner_count += 1
        return V, g

    def policy_iteration(s, env):
        """Docstring."""
        s.V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
        s.V_t = np.empty(env.dim, dtype=np.float32)  # V_{t}
        s.W = np.empty(env.dim_i, dtype=np.float32)
        s.Pi = s.init_pi(env)
        s.Pi = s.Pi.reshape(env.size_i)
        while not s.stable:
            s.V, s.g = s.policy_evaluation(env, s.V, s.W, s.Pi, s.g,
                                           'Policy Evaluation of PI', s.count)
            s.W = s.init_w(env, s.V, s.W)
            s.V = s.V.reshape(env.size)
            s.W = s.W.reshape(env.size_i)
            s.Pi, s.stable = s.policy_improvement(s.V, s.W, s.Pi, env.J, env.D,
                                                  env.gamma, env.KEEP_IDLE,
                                                  env.d_i1, env.d_i2, env.d_f,
                                                  env.P_xy)
            s.V = s.V.reshape(env.dim)
            s.W = s.W.reshape(env.dim_i)
            if s.count > env.max_iter:
                break
            s.count += 1
        s.Pi = s.Pi.reshape(env.dim_i)


class ValueIteration:
    """Value Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    def __init__(self, env, pi_learner):
        self.name = 'Value Iteration'
        self.pi_learner = pi_learner
        self.V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
        self.V_t = np.zeros(env.dim, dtype=np.float32)  # V_{t}
        self.W = np.zeros(env.dim_i, dtype=np.float32)
        self.Pi = pi_learner.init_pi(env)
        self.g = 0
        self.count = 0

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
        for s_i in nb.prange(len(d_i2['s'])):
            for x_i in nb.prange(len(d_i2['x'])):
                for i in nb.prange(J + 1):
                    x = d_i2['x'][x_i]
                    s = d_i2['s'][s_i]
                    state = i * d_i['sizes_i'][0] + np.sum(
                        x * sizes_x + s * sizes_s)
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
                            if w > W[state]:
                                W[state] = w
        return W

    def get_v(self, env, V, W):
        """V_t."""
        states_c = [slice(None)] * (env.J * 2)
        self.V_t = env.tau * V
        for i in range(env.J):
            states_i = np.append(i, [slice(None)] * (env.J * 2))

            states = states_c.copy()
            next_states = states_i.copy()
            states[i] = 0  # x_i = 0
            next_states[1 + i] = 0
            self.V_t[tuple(states)] += env.lab[i] * (W[tuple(next_states)]
                                                     - V[tuple(states)])

            states = states_c.copy()
            next_states = states_i.copy()
            states[i] = slice(1, env.D + 1)  # 0 < x_i <= D
            next_states[1 + i] = slice(1, env.D + 1)  # 0 < x_i <= D
            self.V_t[tuple(states)] += env.gamma * (W[tuple(next_states)]
                                                    - V[tuple(states)])

            for s_i in range(1, env.S + 1):  # s_i
                states = states_c.copy()
                next_states = states_i.copy()
                states[env.J + i] = s_i
                next_states[0] = env.J
                next_states[1 + env.J + i] = s_i - 1
                self.V_t[tuple(states)] += (s_i * env.mu[i]
                                            * (W[tuple(next_states)]
                                               - V[tuple(states)]))
        return self.V_t / env.tau

    def value_iteration(s, env):
        converged = False
        while not converged:  # Update each state.
            s.W = s.pi_learner.init_w(env, s.V, s.W)
            s.V = s.V.reshape(env.size)
            s.W = s.W.reshape(env.size_i)
            s.W = s.get_w(s.V, s.W, env.J, env.D, env.gamma,
                          env.d_i1, env.d_i2, env.d_f, env.P_xy)
            s.V = s.V.reshape(env.dim)
            s.W = s.W.reshape(env.dim_i)
            s.V_t = s.get_v(env, s.V, s.W)
            if s.count % env.convergence_check == 0:
                converged, g = s.pi_learner.convergence(env, s.V_t, s.V,
                                                        s.count, s.name)
            s.V = s.V_t - s.V_t[tuple([0] * (env.J * 2))]  # Rescale V_t
            if s.count > env.max_iter:
                break
            s.count += 1

    def get_policy(s, env):
        """Determine policy via Policy Improvement."""
        s.W = s.init_w(env, s.V, s.W)
        s.V = s.V.reshape(env.size)
        s.W = s.W.reshape(env.size_i)
        s.Pi = s.Pi.reshape(env.size_i)
        s.Pi, _ = s.pi_learner.policy_improvement(s.V, s.W, s.Pi, env.J, env.D,
                                                  env.gamma, env.KEEP_IDLE,
                                                  env.d_i1, env.d_i2, env.d_f,
                                                  env.P_xy)
        s.V = s.V.reshape(env.dim)
        s.W = s.W.reshape(env.dim_i)
        s.Pi = s.Pi.reshape(env.dim_i)

class OneStepPolicyImprovement:
    """One-step policy improvement."""

    def __init__(self, env, pi_learner):
        self.name = 'One-step Policy Improvement'
        self.pi_learner = pi_learner
        self.V_app = self.get_v_app(env)
        self.Pi = pi_learner.init_pi(env)
        self.g = 0

    @staticmethod
    def get_v_app_i(env, i):
        """Calculate V for a single queue."""
        s = env.s_star[i]
        lab = env.lab[i]
        mu = env.mu[i]
        rho = env.rho[i]
        a = env.a[i]
        r = env.r[i]
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
        W = s.pi_learner.init_w(env, s.V_app, s.W)
        s.V_app = s.V_app.reshape(env.size)
        W = W.reshape(env.size_i)
        s.Pi = s.Pi.reshape(env.size_i)
        s.Pi, _ = s.pi_learner.policy_improvement(s.V_app, W, s.Pi, env.J,
                                                  env.D, env.gamma,
                                                  env.KEEP_IDLE, env.d_i1,
                                                  env.d_i2, env.d_f, env.P_xy)
        s.V_app = s.V_app.reshape(env.dim)
        s.Pi = s.Pi.reshape(env.dim_i)

    def get_g(s, env):
        """Determine g via Policy Evaluation."""
        s.g, _ = s.pi_learner.policy_evaluation(env, s.V_app, s.W, s.Pi, s.g,
                                                s.name)
