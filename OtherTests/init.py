"""
Parameter Init

Input variables, Necessary
'J' classes, int, J > 1
'gamma' time discritization parameter, float, gamma > 0
'D' waiting time transitions cap, int, assumed D > 1
'P' penalty never starting service, float, P >= 0
'e' epsilon (small number), float, used for convergence check, 0 < e << 1

Input Programming variables, Necessary
'max_iter' of Value or Policy Iteration (VI and PI), int
'trace' print time and convergence? boolean
'print_modulo' determine when to print

Input variables, Optional
'S' servers, float, S > 0, random [S_MIN, S_MAX]
'mu' service rate, np array, mu > 0, random [mu_MIN, mu_MAX]
'lab(lambda)' arrival rate, np array, lab > 0, based on rho and random weights.
'rho' total system load, float, 0 < rho < 1, rho = lab / (s*mu), random [load_MIN, load_MAX]
't' target time, np.array, t >= 0, t = 1/3
'c' cost, np.array, c > 0, c = 1

Constants
'S_MIN' min No. servers, int, >=1
'S_MAX' max No. servers, int, >=S_MIN
'mu_MIN' min service rate, float, >0
'mu_MAX' max service rate, float, >mu_MIN
'load_MIN' min system load, 0 < rho < 1
'load_MAX' max system load, 0 < rho < 1
'imbalance_MIN' min imbalance between queue loads, float, >0
'imbalance_MAX' max imbalance between queue loads, float, >imbalance_min
'TARGET', time target per class, float 1D-array, >=0

Dependent variables
'alpha' No. gamma transitions allowed to wait, float
'P_xy'(i, x, y) trans. prob. class i, jump time from x to y, float 3D-array
'tau' uniformization constant, float
'dim' (D+1, S+1), +1 to include 0 state, tuple
'size' number of states, int
'sizes' size of every element/submatrix per dimension, float 1D-array
's_states' all valid combinations of S states, int 2D-array
'x_states' all combinations of x states, int 2D-array

Policy , Essential to have the numbers in ascending order
Pi = -3, not evalated
Pi = -2, Servers full
Pi = -1, Keep idle
Pi = 0, No one waiting
Pi = i, take queue i into serves, i={1,...,J}

Penalty is an incentive to take people into service, it is not needed when
there is no decision (servers full)
"""

import numpy as np
from numpy import array, arange, zeros, round, exp, ones, eye, dot
from numpy.random import randint, uniform
from itertools import product
from sys import exit, getsizeof
from timeit import default_timer
from numba import njit

from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy import optimize


class Env:
    """Class docstrings go here."""

    S_MIN: int = 2  # Servers
    S_MAX: int = 6
    mu_MIN = 0.1  # Service Rate
    mu_MAX = 1.
    load_MIN = 0.4  # System load
    load_MAX = 1.
    imbalance_MIN = 1.  # Imbalance
    imbalance_MAX = 5.
    TARGET = array([1], float)  # Target

    NONE_WAITING: int = 0
    KEEP_IDLE: int = -1
    SERVERS_FULL: int = -2
    NOT_EVALUATED: int = -3

    def __init__(s, **kwargs):  # **kwargs: Keyword arguments
        """Create all variables describing the environment."""
        s.J: int = kwargs.get('J')
        s.S: int = kwargs.get('S', randint(s.S_MIN, s.S_MAX + 1))  # [a, b)
        mu = kwargs.get('mu', uniform(s.mu_MIN, s.mu_MAX, s.J))
        if 'lab' in kwargs:
            lab = kwargs.get('lab')
            load: float = sum(lam / mu) / s.S
        else:  # Determine arrival rate based on desired load.
            load = kwargs.get('load', uniform(s.load_MIN, s.load_MAX))
            weight = uniform(s.imbalance_MIN, s.imbalance_MAX, s.J)
            lab = mu * s.S * s.load * weight / sum(weight)
        t = array(kwargs.get('t', np.random.choice(s.TARGET, s.J)), float)
        s.gamma = float(kwargs.get('gamma'))
        if any((t % (1 / s.gamma) != 0) | (t < 1 / s.gamma)):
            t = np.floor(t * s.gamma) / s.gamma
            print("Rounded t down to nearest multiple of 1/gamma.")
        s.lab = array(lab, float)
        s.mu = array(mu, float)
        s.c = array(kwargs.get('c', array([1] * s.J)), float)
        s.r = array(kwargs.get('r', array([1] * s.J)), float)
        s.P: int = kwargs.get('P', max(s.c + s.r) * 10)
        s.D: int = kwargs.get('D')
        s.e = kwargs.get('e', 1e-5)

        s.a = array(lab / mu, float)
        s.s_star = array(s.server_allocation(), float)
        s.rho = array(s.a / s.s_star, float)
        s.pi_0 = s.get_pi_0(s.s_star, s.rho)
        s.tail_prob = s.get_tail_prob(s.s_star, s.rho, s.pi_0, s.gamma*s.t)
        s.cap_prob = s.get_time_cap_prob(s.s_star, s.rho, s.pi_0, s.D)
        s.g = s.get_g_app(s.pi_0, s.tail_prob)
        s.p_xy = s.trans_prob()
        s.tau = array(s.S * max(s.mu) + sum(np.maximum(s.lab, s.gamma)), float)

        s.dim = tuple(np.repeat([s.D + 1, s.S + 1], s.J))
        s.sizes = s.def_sizes(s.dim)
        s.size = np.prod(s.dim)
        s.dim_i = tuple(np.append(s.J + 1, np.repeat([s.D + 1, s.S + 1], s.J)))
        s.sizes_i = s.def_sizes(s.dim_i)
        s.size_i = np.prod(s.dim_i)

        s.max_iter = kwargs.get('max_iter', np.Inf)
        s.trace = kwargs.get('trace', False)
        s.print_modulo = kwargs.get('print_modulo', 1)

        s_states = array(list(product(arange(s.S + 1), repeat=s.J)))
        # Valid states
        s.s_states_v = s_states[np.sum(s_states, axis=1) <= s.S]
        # Action states
        s.s_states = s.s_states_v[np.sum(s.s_states_v, axis=1) < s.S]
        s.x_states = array(list(product(arange(s.D + 1), repeat=s.J)))

        s.feasibility(kwargs.get('time_check', True))
        # if self.trace:
        print("J =", s.J, ", D =", s.D, ", s =", s.S,
              ", gamma =", s.gamma,
              ", (P=", s.P, ")",
              ", load=", round(s.load, 4), '\n',
              "lambda:", round(s.lab, 4), '\n',
              "mu:", round(s.mu, 4), '\n',
              "Target:", round(s.t, 4), '\n',
              "r:", round(s.r, 4), '\n',
              "c:", round(s.c, 4), '\n',
              "s_star:", round(s.s_star, 4), '\n',
              "rho:", round(s.rho, 4), '\n',
              "P(W>D):", s.cap_prob)
        assert s.load < 1, "rho < 1 does not hold"

    def trans_prob(s) -> array:
        """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
        P_xy = np.zeros((s.J, s.D + 1, s.D + 1), float)
        A = np.indices((s.D + 1, s.D + 1))  # x=A[0], y=A[1]
        mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
        for i in range(s.J):
            lab = s.lab[i]
            P_xy[i, 1:, 1:][mask_tril] = ((s.gamma / (lab + s.gamma))
                                          ** (A[0, 1:, 1:][mask_tril]
                                              - A[1, 1:, 1:][mask_tril])
                                          * lab / (lab + s.gamma))
            P_xy[i, 1:, 0] = (s.gamma / (lab + s.gamma)) ** A[0, 1:, 0]
        P_xy[:, 0, 0] = 1
        return P_xy

    def get_pi_0(env, s, rho):
        """Calculate pi(0)."""
        return (1 / (s * exp(s * rho) / (s * rho) ** s
                     * gamma_fun(s) * reg_up_inc_gamma(s, s * rho)
                     + (env.gamma + rho * env.lab) / env.gamma
                     * (1 / (1 - rho))))

    def get_tail_prob(env, s, rho, pi_0, n):
        """P(W>t)."""
        return (pi_0 / (1 - rho) * (env.lab + env.gamma)
                / (env.gamma + env.lab * pi_0)
                * (1 - (s * env.mu - env.lab) / (s * env.mu + env.gamma))
                ** n)

    def get_g_app(s, pi_0, tail_prob):
        return (s.r - s.c * tail_prob) * (s.lab + pi_0 * s.lab ** 2 / s.gamma)

    def server_allocation_cost(env, s):
        """Sums of g per queue, note that -reward is returned."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = env.a / s
            pi_0 = env.get_pi_0(s, rho)
            tail_prob = env.get_tail_prob(s, rho, pi_0)
        tail_prob[~np.isfinite(tail_prob)] = 1  # Correct dividing by 0
        res = env.get_g_app(pi_0, tail_prob)
        return -np.sum(res, axis=len(np.shape(s)) - 1)

    def server_allocation(s):
        """Docstring."""
        weighted_load = s.a / sum(s.a)
        if np.all(s.t > 0):
            weighted_load *= (1 / s.t) / sum((1 / s.t))
        if sum(s.c + s.r) > 0:
            weighted_load *= (s.c + s.r) / sum(s.c + s.r)
        x0 = s.a + weighted_load / sum(weighted_load) * (s.S - sum(s.a))
        lb_bound = s.a  # lb <= A.dot(x) <= ub
        ub_bound = s.S - dot((ones((s.J, s.J)) - eye(s.J)), s.a)
        bounds = optimize.Bounds(lb_bound, ub_bound)
        cons = array([1] * s.J)
        lb_cons = s.S  # Equal bounds represent equality constraint
        ub_cons = s.S
        lin_cons = optimize.LinearConstraint(cons, lb_cons, ub_cons)
        s_star = optimize.minimize(s.server_allocation_cost, x0,
                                   bounds=bounds, constraints=lin_cons).x
        return s_star

    @staticmethod
    def def_sizes(dim):
        """Docstring."""
        sizes = np.zeros(len(dim), int)
        sizes[-1] = 1
        for i in range(len(dim) - 2, -1, -1):
            sizes[i] = sizes[i + 1] * dim[i + 1]
        return sizes

    def init_Pi(s):
        """
        Take the longest waiting queue into service (or last queue if tied).
        Take arrivals directly into service.
        """
        Pi = s.NOT_EVALUATED * np.ones(s.dim_i, dtype=int)
        for s in s.s_states_v:
            states = np.append([slice(None)] * (1 + s.J), s)
            if np.sum(s) == s.S:
                Pi[tuple(states)] = s.SERVERS_FULL
                continue
            for i in arange(s.J):
                states_ = states.copy()
                for x in arange(1, s.D + 1):
                    states_[1 + i] = x  # x_i = x
                    for j in arange(s.J):
                        if j != i:
                            states_[1 + j] = slice(0, x + 1)  # 0 <= x_j <= x_i
                    Pi[tuple(states_)] = i + 1
                states_ = states.copy()
                states_[0] = i
                states_[1 + i] = 0
                Pi[tuple(states)] = i + 1  # Admit arrival (of i)
            states = np.concatenate(([s.J], [0] * s.J, s),
                                    axis=0)  # x_i = 0 All i
            Pi[tuple(states)] = s.NONE_WAITING
        return Pi

    def timer(self, start: bool, name: str, trace: bool):
        """Only if trace=TRUE, start timer if start=true, else print time."""
        if not trace:
            pass
        elif start:
            print('Starting ' + name + '.')
            self.start = default_timer()
        else:
            time = default_timer() - self.start
            print("Time: ", int(time / 60), ":",
                  int(time - 60 * int(time / 60)))
            print('Finished ' + name + '.')
            return time

    @staticmethod
    @njit
    def test_loop(memory, size_i, sizes_i, s_states, x_states, J):
        """Docstring."""
        memory = memory.reshape(size_i)
        for s in s_states:
            for x in x_states:
                for i in arange(J + 1):
                    state = i * sizes_i[0] + np.sum(
                        x * sizes_i[1:J + 1] + s * sizes_i[J + 1:J * 2 + 1])
                    memory[state] = np.random.rand()

    def feasibility(self, time_check):
        """Check matrix size and looping time."""
        memory = zeros(self.dim_i)
        name = 'Running time feasibility'
        if self.trace:
            print("Size of W: ", round(getsizeof(memory) / 10 ** 9, 4), "GB.")
            print("Size of V: ", round(getsizeof(zeros(self.dim)) / 10 ** 9, 4),
                  "GB.")
        if time_check:
            self.timer(True, name, True)
            s.test_loop(memory, self.size_i, self.sizes_i, self.s_states,
                           self.x_states, self.J)
            time = self.timer(False, name, True)
            if time > 60:  # in seconds
                exit("Looping matrix takes more than 60 seconds.")

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
        if ((converged & self.trace) |
                (self.trace & (i % self.print_modulo == 0 |
                               j % self.print_modulo == 0))):
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
