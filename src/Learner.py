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
from numpy import array
import numba as nb
from numba import types as tp
import Env
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy.integrate import quad_vec
# from numpy import array, arange, zeros, eye, ones, dot, exp, around
# from scipy import optimize


class PolicyIteration:
    """Policy Iteration."""
    DICT_TYPE_I1 = tp.DictType(tp.unicode_type, tp.i4[:])  # int 1D vector
    DICT_TYPE_I2 = tp.DictType(tp.unicode_type, tp.i4[:, :])  # int 2D vector
    DICT_TYPE_F = tp.DictType(tp.unicode_type, tp.f8[:])  # float 1D vector

    def __init__(self, env):
        self.name = 'Policy Iteration'
        self.V = np.zeros(env.dim, dtype=np.float32)  # V_{t-1}
        self.W = np.zeros(env.dim_i, dtype=np.float32)
        self.Pi = env.init_pi()
        self.g = 0
        self.count = 0
        self.stable = False

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

    def policy_evaluation(self, env, V, W, Pi, name, count=0):
        """Policy Evaluation."""
        inner_count = 0
        converged = False
        while not converged:
            W = env.init_w(V, W)
            V = V.reshape(env.size)
            W = W.reshape(env.size_i)
            W = self.get_w(V, W, Pi, env.J, env.D, env.gamma,
                           self.d_i1, self.d_i2, self.d_f, env.P_xy)
            V = V.reshape(env.dim)
            W = W.reshape(env.dim_i)
            V_t = self.get_v(env, V, W)
            if inner_count % env.convergence_check == 0:
                converged, g = env.convergence(V_t, V, count, name,
                                               j=inner_count)
            V = V_t - V_t[tuple([0] * (env.J * 2))]  # Rescale and Save V_t
            if count > env.max_iter:
                return V, g
            inner_count += 1
        return V, g

    def policy_iteration(self, env):
        """Docstring."""
        self.Pi = self.Pi.reshape(env.size_i)
        env.timer(True, self.name, env.trace)
        while not self.stable:
            V, g = self.policy_evaluation(env, V, W, Pi,
                                          'Policy Evaluation of PI', self.count)
            W = env.init_w(V, W)
            V = V.reshape(env.size)
            W = W.reshape(env.size_i)
            Pi, stable = self.policy_improvement(V, W, Pi, env.J, env.D,
                                                 env.gamma, env.KEEP_IDLE,
                                                 self.d_i1, self.d_i2, self.d_f,
                                                 env.P_xy)
            V = V.reshape(env.dim)
            W = W.reshape(env.dim_i)
            if self.count > env.max_iter:
                break
            self.count += 1
        env.timer(False, self.name, env.trace)
        self.Pi = self.Pi.reshape(env.dim_i)


class ValueIteration():
    """Value Iteration."""

    def __init__(self, env, PI_learner):
        self.name = 'Value Iteration'
        self.V = zeros(env.dim)  # V_{t-1}
        self.W = zeros(env.dim)
        self.Pi = PI_learner.initialize_pi(env)
        self.g = 0
        self.count = 0

    @staticmethod
    @njit
    def W_f(V, W, J, S, D, gamma, t, c, r, P, size, sizes, S_states, x_states, dim, P_xy):
        """W."""
        V = V.reshape(size);
        W = W.reshape(size)
        for s in S_states:
            for x in x_states:
                state = np.sum(x * sizes[0:J] + s * sizes[J:J * 2])
                W[state] = V[state]
                if np.sum(s) < S:
                    W[state] = W[state] - P if np.any(x == D) else W[state]
                    for i in arange(J):
                        if (x[i] > 0):  # If someone of class i waiting
                            value = r[i] - c[i] if x[i] > gamma * t[i] else r[i]
                            for y in arange(x[i] + 1):
                                next_x = x.copy()
                                next_x[i] = y
                                next_s = s.copy()
                                next_s[i] += 1
                                next_state = np.sum(next_x * sizes[0:J] + \
                                                    next_s * sizes[J:J * 2])
                                value += P_xy[i, x[i], y] * V[next_state]
                            W[state] = array([value, W[state]]).max()
        return W.reshape(dim)

    def value_iteration(self, env, PI_learner):
        """Docstring."""
        env.timer(True, self.name, env.trace)
        while True:  # Update each state.
            self.W = self.W_f(self.V, self.W, env.J, env.S, env.D, env.gamma,
                              env.t, env.c, env.r, env.P, env.size, env.sizes,
                              env.S_states, env.x_states, env.dim, env.P_xy)
            V_t = PI_learner.V_f(env, self.V, self.W)
            converged, self.g = PI_learner.convergence(env, V_t, self.V, self.count, self.name)
            if (converged):
                break  # Stopping condition
            # Rescale and Save V_t
            self.V = V_t - V_t[tuple([0] * (env.J * 2))]
            self.count += 1
        env.timer(False, self.name, env.trace)

    def policy(self, env, PI_learner):
        """Determine policy via Policy Improvement."""
        self.Pi, _ = PI_learner.policy_improvement(self.V, self.Pi, True,
                                                   env.J, env.S, env.D, env.gamma, env.t, env.c, env.r, env.P,
                                                   env.size, env.sizes, env.S_states, env.x_states, env.dim, env.P_xy,
                                                   PI_learner.NONE_WAITING, PI_learner.KEEP_IDLE,
                                                   PI_learner.SERVERS_FULL)

    def convergence(self, V_t, V, i, name, j=-1):
        """Convergence check of valid states only."""
        delta_max = V_t[tuple([0] * (self.J * 2))] - V[
            tuple([0] * (self.J * 2))]
        delta_min = delta_max.copy()
        for s in self.s_states_v:
            states = [slice(None)] * (self.J * 2)
            states[slice(self.J, self.J * 2)] = s
            diff = V_t[tuple(states)] - V[tuple(states)]
            delta_max = np.max([np.max(diff), delta_max])
            delta_min = np.min([np.min(diff), delta_min])
            if abs(delta_max - delta_min) > self.e:
                break
        converged = delta_max - delta_min < self.e
        max_iter = (i > self.max_iter) | (j > self.max_iter)
        g = (delta_max + delta_min) / 2 * self.tau
        if ((converged & self.trace)
                | (self.trace & (((i % self.print_modulo == 0) & (j == -1))
                                 | (j % self.print_modulo == 0)))):
            print("iter: ", i,
                  "inner_iter: ", j,
                  ", delta: ", round(delta_max - delta_min, 2),
                  ', D_min', round(delta_min, 2),
                  ', D_max', round(delta_max, 2),
                  ", g: ", round(g, 4))
        elif converged:
            if j == -1:
                print(name, 'converged in', i, 'iterations. g=',
                      round(g, 4))
            else:
                print(name, 'converged in', j, 'iterations. g=',
                      round(g, 4))
        elif max_iter:
            print(name, 'iter:', i, 'reached max_iter =', max_iter, ', g~',
                  round(g, 4))
        return converged | max_iter, g


class OneStepPolicyImprovement():
    """Doing one step policy improvement."""

    def __init__(self, env, PI_learner):
        self.name = 'One-step Policy Improvement'
        self.V = zeros(env.dim)  # V_{t-1}
        self.W = zeros(env.dim)
        self.Pi = PI_learner.initialize_pi(env)
        self.g = 0
        self.s_star = self.server_allocation(env)
        self.rho = env.a / self.s_star
        print(self.s_star, self.rho)
        self.pi_0 = self.get_pi_0(env, self.s_star, self.rho)
        self.tail_prob = self.get_tail_prob(env, self.s_star, self.rho, self.pi_0)
        self.g_app = self.get_g_app(env, self.pi_0, self.tail_prob)

        if env.trace:
            print("s_star:", around(self.s_star, 4), '\n',
                  "rho:", around(self.rho, 4))

    def get_pi_0(self, env, s, rho):
        """Calculate pi(0)."""
        pi_0 = s * exp(s * rho) / (s * rho) ** s * \
               gamma_fun(s) * reg_up_inc_gamma(s, s * rho)
        pi_0 += (env.gamma + rho * env.lmbda) / env.gamma * (1 / (1 - rho))
        return 1 / pi_0

    def get_tail_prob(self, env, s, rho, pi_0):
        """P(W>t)."""
        tail_prob = pi_0 / (1 - rho) * \
                    (env.lmbda + env.gamma) / (env.gamma + env.lmbda * pi_0) * \
                    (1 - (s * env.mu - env.lmbda) / (s * env.mu + env.gamma)
                     ) ** (env.gamma * env.t)
        return tail_prob

    def get_g(self, env, pi_0, tail_prob):
        """g for every isolated queue."""
        return (env.r - env.c * tail_prob) * \
               (env.lmbda + pi_0 * env.lmbda ** 2 / env.gamma)

    def server_allocation_cost(self, s, env):
        """Sums of g per queue, note that -reward is returned."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = env.a / s
            pi_0 = self.get_pi_0(env, s, rho)
            tail_prob = self.get_tail_prob(env, s, rho, pi_0)
        tail_prob[~np.isfinite(tail_prob)] = 1  # Correct dividing by 0
        res = self.get_g_app(env, pi_0, tail_prob)
        return -np.sum(res, axis=len(np.shape(s)) - 1)

    def server_allocation(self, env):
        """Docstring."""
        if np.all(env.t > 0):
            weighted_load = (1 / env.t) / sum((1 / env.t)) * \
                            env.c / sum(env.c) * env.a / sum(env.a)
        else:
            weighted_load = env.c / sum(env.c) * env.a / sum(env.a)
        x0 = env.a + weighted_load / sum(weighted_load) * (env.S - sum(env.a))
        lb_bound = env.a  # lb <= A.dot(x) <= ub
        ub_bound = env.S - dot((ones((env.J, env.J)) - eye(env.J)), env.a)
        bounds = optimize.Bounds(lb_bound, ub_bound)

        A_cons = array([1] * env.J)
        lb_cons = env.S  # Equal bounds represent equality constraint
        ub_cons = env.S
        lin_cons = optimize.LinearConstraint(A_cons, lb_cons, ub_cons)

        s_star = optimize.minimize(self.server_allocation_cost, x0, args=(env),
                                   bounds=bounds, constraints=lin_cons).x
        return s_star

    def V_app_f(self, env, i):
        """Calculate V for a single queue."""
        s = self.s_star[i];
        lmbda = env.lmbda[i];
        mu = env.mu[i];
        rho = self.rho[i];
        a = env.a[i];
        r = env.r[i];
        c = env.c[i];
        t = env.t[i];
        g = self.g_app[i]
        V_i = zeros(env.D + 1)

        # V(x) for x<=0, with V(-s)=0
        V_x_le_0 = lambda y: (1 - (y / a) ** (s)) / (1 - y / a) * exp(a - y)
        V_i[0] = (g - lmbda * r) / lmbda * quad_vec(V_x_le_0, a, np.inf)[0]

        # V(x) for x>0
        x = arange(1, env.D + 1)
        frac = (s * mu + env.gamma) / (lmbda + env.gamma)
        trm = exp(a) / a ** (s - 1) * gamma_fun(s) * reg_up_inc_gamma(s, a)
        V_i[x] = V_i[0] + (s * mu * r - g) / (env.gamma * s * mu * (1 - rho) ** 2) * \
                 (lmbda + env.gamma - lmbda * x * (rho - 1) - (lmbda + env.gamma) * frac ** x) + \
                 1 / (env.gamma * (rho - 1)) * (g - s * mu * r - env.gamma / lmbda * (
                g + (g - lmbda * r) / rho * trm)) * (-rho + frac ** (x - 1))
        # -1_{x > gamma*t}[...]
        x = arange(env.gamma * t + 1, env.D + 1).astype(int)
        V_i[x] -= c / (env.gamma * (1 - rho) ** 2) * \
                  (lmbda + env.gamma - lmbda * (x - env.gamma * t - 1) * \
                   (rho - 1) - (lmbda + env.gamma) * frac ** (x - env.gamma * t - 1))
        return V_i

    def get_V_app(self, env, V_app):
        """Approximation of value function.
    
        Create a list V_memory with V_ij(x), i=class, j=#servers for all x.
        Note only j = s*_i, ..., s will be filled, rest zero
        """
        for i in arange(env.J):
            V_i = self.V_app_f(env, i)
            for x in arange(env.D + 1):
                states = [slice(None)] * (env.J * 2)
                states[i] = x
                V_app[tuple(states)] += V_i[x]
        return V_app

    def one_step_policy_improvement(self, env, PI_learner):
        """One Step of Policy Improvement."""
        env.timer(True, self.name, env.trace)
        self.V = self.get_V_app(env, self.V)
        self.Pi, _ = PI_learner.policy_improvement(self.V, self.Pi, True,
                                                   env.J, env.S, env.D, env.gamma, env.t, env.c, env.r, env.P,
                                                   env.size, env.sizes, env.S_states, env.x_states, env.dim, env.P_xy,
                                                   PI_learner.NONE_WAITING, PI_learner.KEEP_IDLE,
                                                   PI_learner.SERVERS_FULL)
        env.timer(False, self.name, env.trace)

    def calculate_g(self, env, PI_learner):
        """Determine g and policy via Policy Improvement."""
        self.g, self.V, self.W = PI_learner.policy_evaluation(env, self.V, self.W, self.Pi, self.name)
