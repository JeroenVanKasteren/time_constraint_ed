"""
MDP environment class.

All elements of 'self':

Input variables, Necessary
'J' classes, int, J > 1
'gamma' time discritization parameter, float, gamma > 0
'D' waiting time transitions cap, int, assumed D > 1
'P' penalty never starting service, flaot, P >= 0
'e' epsilon (small number), float, used for convergence check, 0 < e << 1

Input Programming variables, Necessary
'max_iter' of Value or Policy Iteration (VI and PI), int
'print_modulo' determine when to print

Input variables, Optional
'S' servers, float, S > 0, random [S_MIN, S_MAX]
'mu' service rate, np array, mu > 0, random [mu_MIN, mu_MAX]
'lambda' arrival rate, np array, lambda > 0, based on rho and random weights.
'rho' total system load, float, 0 < rho < 1, rho = lambda / (s*mu),
random [load_MIN, load_MAX]
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
'S_states' all valid combinations of S states, int 2D-array
'x_states' all combinations of x states, int 2D-array

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
import pandas as pd
from numpy import array, round, int32
from numpy.random import randint
from itertools import product
from sys import getsizeof as size
import numba as nb
from numba import types as tp
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma
from scipy import optimize
from time import perf_counter as clock, strptime


class TimeConstraintEDs:
    """Class docstrings go here."""

    S_MIN: int = 2  # Servers
    S_MAX: int = 10
    mu_MIN = 0.2  # Service Rate
    mu_MAX = 1.
    load_MIN = 0.4  # System load
    load_MAX = 0.9
    imbalance_MIN = 1.  # Imbalance
    imbalance_MAX = 5.
    TARGET = array([1], float)  # Target

    ZERO_ONE_PERC = 1e-3
    NONE_WAITING: int = 0
    KEEP_IDLE: int = -1
    SERVERS_FULL: int = -2
    NOT_EVALUATED: int = -3

    def __init__(s, **kwargs):  # **kwargs: Keyword arguments
        s.rng = np.random.default_rng(kwargs.get('seed', 42))
        s.J: int = np.int64(kwargs.get('J'))
        s.S: int = np.int64(kwargs.get('S',
                                       s.rng.integers(s.S_MIN, s.S_MAX + 1)))
        mu = kwargs.get('mu', s.rng.uniform(s.mu_MIN, s.mu_MAX, s.J))
        if 'lab' in kwargs:
            lab = kwargs.get('lab')
            s.load: float = sum(lab / mu) / s.S
        elif 'imbalance' in kwargs:
            s.imbalance = kwargs.get('imbalance')
            s.load = kwargs.get('load', s.rng.uniform(s.load_MIN, s.load_MAX))
            lab = mu * s.S * s.load * s.imbalance / sum(s.imbalance)
        else:
            s.load = kwargs.get('load', s.rng.uniform(s.load_MIN, s.load_MAX))
            weight = s.rng.uniform(s.imbalance_MIN, s.imbalance_MAX, s.J)
            lab = mu * s.S * s.load * weight / sum(weight)
        t = array(kwargs.get('t', s.rng.choice(s.TARGET, s.J)), float)
        s.gamma = float(kwargs.get('gamma'))

        s.trace = kwargs.get('trace', False)
        if any((t % (1 / s.gamma) != 0) | (t < 1 / s.gamma)):
            t = np.floor(t * s.gamma) / s.gamma
            if s.trace:
                print('Rounded t down to nearest multiple of 1/gamma.\n')
        s.lab = array(lab, float)
        s.mu = array(mu, float)
        s.t = array(t, float)
        s.c = array(kwargs.get('c', array([1] * s.J)), float)
        s.r = array(kwargs.get('r', array([1] * s.J)), float)
        s.P: int = int(kwargs.get('P', max(s.c + s.r) * 10))
        s.e = kwargs.get('e', 1e-5)

        s.a = array(lab / mu, float)
        s.s_star = array(s.server_allocation(), float)
        s.rho = array(s.a / s.s_star, float)
        s.pi_0 = s.get_pi_0(s.s_star, s.rho, s.lab)
        s.tail_prob = s.get_tail_prob(s.s_star, s.rho, s.lab, s.mu,
                                      s.pi_0, s.gamma * s.t)
        s.g = s.get_g_app(s.pi_0, s.tail_prob)
        s.tau = float(s.S * max(s.mu) + sum(np.maximum(s.lab, s.gamma)))

        if 'D' in kwargs:
            s.D: int = kwargs.get('D')
        else:
            s.D: int = s.get_D()
        s.cap_prob_i = s.get_tail_prob(s.s_star, s.rho,
                                       s.lab, s.mu, s.pi_0, s.D)
        mu = sum(s.lab) / sum(s.lab / s.mu)
        pi_0 = s.get_pi_0(s.S, s.load, sum(s.lab))
        s.cap_prob = s.get_tail_prob(s.S, s.load, sum(s.lab), mu, pi_0, s.D)
        s.target_prob = s.get_tail_prob(s.S, s.load, sum(s.lab), mu, pi_0,
                                        max(s.t))

        s.P_xy = s.trans_prob()

        s.dim = tuple(np.repeat([s.D + 1, s.S + 1], s.J))
        s.sizes = s.def_sizes(s.dim)
        s.size = np.prod(s.dim)
        s.dim_i = tuple(np.append(s.J + 1, np.repeat([s.D + 1, s.S + 1], s.J)))
        s.sizes_i = s.def_sizes(s.dim_i)
        s.size_i = np.prod(s.dim_i).astype(int32)

        s.max_iter = kwargs.get('max_iter', np.Inf)  # max(size_i^2, 1e3)
        s.start_time = clock()
        s.max_time = s.get_time(kwargs.get('max_time', None))

        s.print_modulo = kwargs.get('print_modulo', 1e10)  # 1 for always
        s.convergence_check = kwargs.get('convergence_check', 1)

        s_states = array(list(product(np.arange(s.S + 1), repeat=s.J)), int32)
        # Valid states
        s.s_states_v = s_states[np.sum(s_states, axis=1) <= s.S]
        # Action states
        s.s_states = s.s_states_v[np.sum(s.s_states_v, axis=1) < s.S]
        s.s_states_full = s.s_states_v[np.sum(s.s_states_v, axis=1) == s.S]
        s.x_states = array(list(product(np.arange(s.D + 1), repeat=s.J)), int32)

        s.d_i0 = nb.typed.Dict.empty(key_type=tp.unicode_type,
                                     value_type=tp.i8)
        s.d_i0['J'] = s.J
        s.d_i0['D'] = s.D
        s.d_i0['S'] = np.int64(s.S)
        s.d_i0['P'] = s.P
        s.d_i0['convergence_check'] = s.convergence_check
        s.d_i0['print_modulo'] = np.int64(s.print_modulo)
        s.d_i1 = nb.typed.Dict.empty(key_type=tp.unicode_type,
                                     value_type=tp.i4[:])
        s.d_i1['sizes'] = s.sizes
        s.d_i1['sizes_i'] = s.sizes_i
        s.d_i1['P_m'] = s.get_P_m()
        s.d_i2 = nb.typed.Dict.empty(key_type=tp.unicode_type,
                                     value_type=tp.i4[:, :])
        s.d_i2['s'] = s.s_states
        s.d_i2['s_valid'] = s.s_states_v
        s.d_i2['x'] = s.x_states
        s.d_f0 = nb.typed.Dict.empty(key_type=tp.unicode_type,
                                     value_type=tp.f8)
        s.d_f0['max_iter'] = float(s.max_iter)
        s.d_f0['start_time'] = float(s.start_time)
        s.d_f0['max_time'] = float(s.max_time)
        s.d_f0['tau'] = s.tau
        s.d_f0['e'] = s.e
        s.d_f1 = nb.typed.Dict.empty(key_type=tp.unicode_type,
                                     value_type=tp.f8[:])
        s.d_f1['t'] = s.t
        s.d_f1['c'] = s.c
        s.d_f1['r'] = s.r
        s.d_f1['lab'] = s.lab
        s.d_f1['mu'] = s.mu

        if s.trace:
            print(f'J = {s.J} D = {s.D}, s = {s.S}, gamma = {s.gamma},'
                  f'P = {s.P} \n'
                  f'load = {s.load:.4f}\n'
                  f'lambda = {round(s.lab, 4)}\n'
                  f'mu = {round(s.mu, 4)}\n'
                  f'target = {round(s.t, 4)}\n'
                  f'r = {s.r}\n'
                  f'c = {s.c}\n'
                  f's_star = {round(s.s_star, 4)}\n'
                  f'rho: {round(s.rho, 4)}\n'
                  f'P(W>t): {s.target_prob}\n'
                  f'P(W>D): {s.cap_prob}\n'
                  f'P(W>D_i): {s.cap_prob_i}\n'
                  f'size: {s.size_i}\n'
                  f'W: {size(np.zeros(s.dim_i, dtype=np.float32)) /10**9:.4f}'
                  f' GB.\n'
                  f'V: {size(np.zeros(s.dim, dtype=np.float32)) /10**9:.4f}'
                  f' GB.\n')
        assert s.load < 1, 'rho < 1 does not hold'

    def get_D(self):
        lab = sum(self.lab)
        mu = lab / sum(self.lab / self.mu)
        pi_0 = self.get_pi_0(self.S, self.load, lab)
        prob_delay = self.get_tail_prob(self.S, self.load, lab, mu, pi_0, 0)
        D = int(np.ceil(-np.log(self.ZERO_ONE_PERC / prob_delay) /
                        (self.S * mu - lab) * self.gamma))
        return D

    def trans_prob(self):
        """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
        P_xy = np.zeros((self.J, self.D + 1, self.D + 1))
        gamma = self.gamma
        A = np.indices((self.D + 1, self.D + 1))  # x=A[0], y=A[1]
        mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
        for i in range(self.J):
            lab = self.lab[i]
            P_xy[i, 1:, 1:][mask_tril] = (gamma / (lab + gamma)) ** \
                                         (A[0, 1:, 1:][mask_tril] -
                                          A[1, 1:, 1:][mask_tril]) * \
                                         lab / (lab + gamma)
            P_xy[i, 1:, 0] = (gamma / (lab + gamma)) ** A[0, 1:, 0]
        P_xy[:, 0, 0] = 1
        return P_xy

    def get_pi_0(self, s, rho, lab):
        """Calculate pi(0)."""
        return (1 / (s * np.exp(s * rho) / (s * rho) ** s
                     * gamma_fun(s) * reg_up_inc_gamma(s, s * rho)
                     + (self.gamma + rho * lab) / self.gamma
                     * (1 / (1 - rho))))

    def get_tail_prob(env, s, rho, lab, mu, pi_0, n):
        """P(W>t)."""
        return (pi_0 / (1 - rho) * (lab + env.gamma)
                / (env.gamma + lab * pi_0)
                * (1 - (s * mu - lab) / (s * mu + env.gamma)) ** n)

    def get_g_app(s, pi_0, tail_prob):
        return (s.r - s.c * tail_prob) * (s.lab + pi_0 * s.lab ** 2 / s.gamma)

    def server_allocation_cost(env, s):
        """Sums of g per queue, note that -reward is returned."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = env.a / s
            pi_0 = env.get_pi_0(s, rho, env.lab)
            tail_prob = env.get_tail_prob(s, rho, env.lab, env.mu,
                                          pi_0, env.gamma * env.t)
        tail_prob[~np.isfinite(tail_prob)] = 1  # Correct dividing by 0
        pi_0[~np.isfinite(tail_prob)] = 0  # Correct dividing by 0
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
        lb_bound = s.a + s.ZERO_ONE_PERC  # lb <= A.dot(x) <= ub
        ub_bound = s.S - np.dot((np.ones((s.J, s.J)) - np.eye(s.J)), s.a)
        bounds = optimize.Bounds(lb_bound, ub_bound)
        cons = array([1] * s.J)
        lb_cons = s.S  # Equal bounds represent equality constraint
        ub_cons = s.S
        lin_cons = optimize.LinearConstraint(cons, lb_cons, ub_cons)
        s_star = optimize.minimize(s.server_allocation_cost, x0,
                                   bounds=bounds, constraints=lin_cons).x
        return s_star

    def def_sizes(self, dim):
        """Docstring."""
        sizes = np.zeros(len(dim), int32)
        sizes[-1] = 1
        for i in range(len(dim) - 2, -1, -1):
            sizes[i] = sizes[i + 1] * dim[i + 1]
        return sizes

    def get_P_m(self):
        """
        Make a matrix with Penalty P.
        """
        P_m = np.zeros(self.dim, dtype=np.int32)
        states = [slice(None)] * (1 + self.J * 2)
        for i in range(self.J):
            states[i] = slice(int(self.gamma * self.t[i]) + 1, self.D + 1)
        for s in self.s_states:
            states[self.J:] = s
            P_m[tuple(states)] = self.P
        return P_m.reshape(self.size)

    @staticmethod
    def get_time(time_string):
        """Read in time in formats (D)D-HH:MM:SS, (H)H:MM:SS, or (M)M:SS."""
        if (time_string is not None) & (time_string is not np.nan) & \
                (time_string):
            if '-' in time_string:
                days, time = time_string.time.split('-')
            elif time_string.count(':') == 1:
                days, time = 0, '0:'+time_string
            else:
                days, time = 0, time_string
            x = strptime(time, '%H:%M:%S')
            return (((int(days) * 24 + x.tm_hour) * 60 + x.tm_min) * 60
                    + x.tm_sec - 60)
        else:
            return np.Inf

    def time_print(self, time):
        """Convert seconds to readable format."""
        print(f'Time: {time/60:.0f}:{time - 60 * int(time / 60):.0f} min.\n')
