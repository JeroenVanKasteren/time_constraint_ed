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
'lambda' arrival rate, np array, lambda > 0, based on rho and random weights.
'rho' total system load, float, 0 < rho < 1, rho = lambda / (s*mu), random [load_MIN, load_MAX]
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
from numpy import array, arange, zeros, round, exp, ones, eye, dot, int, float32
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
    mu_MIN: float32 = float32(0.1)  # Service Rate
    mu_MAX: float32 = float32(1.)
    load_MIN: float32 = float32(0.4)  # System load
    load_MAX: float32 = float32(1)
    imbalance_MIN: float32 = float32(1)  # Imbalance
    imbalance_MAX: float32 = float32(5)
    TARGET = array([1], float32)  # Target

    NONE_WAITING = np.int(0)
    KEEP_IDLE = np.int(-1)
    SERVERS_FULL = np.int(-2)
    NOT_EVALUATED = np.int(-3)

    def __init__(self, **kwargs):  # **kwargs: Keyword arguments
        """Create all variables describing the environment."""
        self.J = np.int(kwargs.get('J'))
        self.S = np.int(kwargs.get('S', randint(self.S_MIN,
                                                self.S_MAX + 1)))  # [a, b)
        self.mu = array(kwargs.get('mu', uniform(self.mu_MIN, self.mu_MAX,
                                                 self.J)), float32)
        if 'lmbda' in kwargs:
            lmbda = kwargs.get('lmbda')
            Rho = array(sum(self.lmbda / self.mu) / self.S, float32)
        else:  # Determine arrival rate based on desired load.
            Rho = kwargs.get('Rho', uniform(self.load_MIN, self.load_MAX))
            weight = uniform(self.imbalance_MIN, self.imbalance_MAX, self.J)
            lmbda = self.mu * self.S * self.Rho * weight / sum(weight)
        self.Rho = array(Rho, float32)
        self.lmbda = array(lmbda, float32)
        t = kwargs.get('t', np.random.choice(self.TARGET, self.J))
        self.gamma = float32(kwargs.get('gamma'))
        if any((t % (1 / self.gamma) != 0) | (t < 1 / self.gamma)):
            t = np.floor(t * self.gamma) / self.gamma
            print("Rounded t down to nearest multiple of 1/gamma.")
        self.t = array(t, float32)
        self.c = array(kwargs.get('c', array([1] * self.J)), float32)
        self.r = array(kwargs.get('r', array([1] * self.J)), float32)
        self.P = float32(kwargs.get('P', max(self.c + self.r) * 10))
        self.D = np.int(kwargs.get('D'))
        self.e = float32(kwargs.get('e', 1e-5))

        self.a = array(self.lmbda / self.mu, float32)
        self.s_star = array(self.server_allocation(), float32)
        self.rho = array(self.a / self.s_star, float32)
        self.pi_0 = self.get_pi_0(self.s_star, self.rho)
        self.tail_prob = self.get_tail_prob(self.s_star, self.rho, self.pi_0)
        self.cap_prob = self.get_time_cap_prob(self.s_star, self.rho, self.pi_0)
        self.g = self.get_g_app(self.pi_0, self.tail_prob)

        self.p_xy = self.trans_prob()
        self.tau = array(self.S * max(self.mu) + sum(
            np.maximum(self.lmbda, self.gamma)), float32)
        self.dim = tuple(array(np.repeat([self.D+1, self.S+1], self.J), int))
        self.sizes = self.def_sizes(self.dim)
        self.size = np.prod(self.dim)
        self.dim_i = tuple(
            np.append(self.J + 1, np.repeat([self.D + 1, self.S + 1], self.J)))
        self.sizes_i = self.def_sizes(self.dim_i)
        self.size_i = np.prod(self.dim_i)

        self.max_iter = kwargs.get('max_iter', np.Inf)
        self.trace = kwargs.get('trace', False)
        self.print_modulo = kwargs.get('print_modulo', 1)

        self.s_states = array(list(product(arange(self.S + 1), repeat=self.J)))
        self.s_states_v = self.s_states[
            np.sum(self.s_states, axis=1) <= self.S]  # Valid states
        self.s_states = self.s_states_v[
            np.sum(self.s_states_v, axis=1) < self.S]  # Action states
        self.x_states = array(list(product(arange(self.D + 1), repeat=self.J)))

        self.feasibility(kwargs.get('time_check', True))
        # if self.trace:
        print("J =", self.J, ", D =", self.D, ", s =", self.S,
              ", gamma =", self.gamma,
              ", (P=", self.P, ")",
              ", Rho=", round(self.Rho, 4), '\n',
              "lambda:", round(self.lmbda, 4), '\n',
              "mu:", round(self.mu, 4), '\n',
              "Target:", round(self.t, 4), '\n',
              "r:", round(self.r, 4), '\n',
              "c:", round(self.c, 4), '\n',
              "s_star:", round(self.s_star, 4), '\n',
              "rho:", round(self.rho, 4), '\n',
              "P(W>D):", self.cap_prob)
        assert self.Rho < 1, "rho < 1 does not hold"

    def trans_prob(self) -> array:
        """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
        P_xy = np.zeros((self.J, self.D + 1, self.D + 1), np.float32)
        gamma = self.gamma
        A = np.indices((self.D + 1, self.D + 1), dtype=int)  # x=A[0], y=A[1]
        mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
        for i in range(self.J):
            lmbda = self.lmbda[i]
            P_xy[i, 1:, 1:][mask_tril] = ((gamma / (lmbda + gamma))
                                          ** (A[0, 1:, 1:][mask_tril]
                                              - A[1, 1:, 1:][mask_tril])
                                          * lmbda / (lmbda + gamma))
            P_xy[i, 1:, 0] = (gamma / (lmbda + gamma)) ** A[0, 1:, 0]
        P_xy[:, 0, 0] = 1
        return P_xy

    def get_pi_0(self, s, rho):
        """Calculate pi(0)."""
        return (1 / (s * exp(s * rho) / (s * rho) ** s
                     * gamma_fun(s) * reg_up_inc_gamma(s, s * rho)
                     + (self.gamma + rho * self.lmbda) / self.gamma
                     * (1 / (1 - rho))))

    def get_tail_prob(self, s, rho, pi_0):
        """P(W>t)."""
        return (pi_0 / (1 - rho) * (self.lmbda + self.gamma)
                / (self.gamma + self.lmbda * pi_0)
                * (1 - (s * self.mu - self.lmbda) / (s * self.mu + self.gamma))
                ** (self.gamma * self.t))

    def get_time_cap_prob(self, s, rho, pi_0):
        """P(W>D)."""
        return (pi_0 / (1 - rho) * (self.lmbda + self.gamma)
                / (self.gamma + self.lmbda * pi_0)
                * (1 - (s * self.mu - self.lmbda) / (s * self.mu + self.gamma))
                ** self.D)

    def get_g_app(self, pi_0, tail_prob):
        return ((self.r - self.c * tail_prob)
                * (self.lmbda + pi_0 * self.lmbda ** 2 / self.gamma))

    def server_allocation_cost(self, s):
        """Sums of g per queue, note that -reward is returned."""
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = self.a / s
            pi_0 = self.get_pi_0(s, rho)
            tail_prob = self.get_tail_prob(s, rho, pi_0)
        tail_prob[~np.isfinite(tail_prob)] = 1  # Correct dividing by 0
        res = self.get_g_app(pi_0, tail_prob)
        return -np.sum(res, axis=len(np.shape(s)) - 1)

    def server_allocation(self):
        """Docstring."""
        weighted_load = self.a / sum(self.a)
        if np.all(self.t > 0):
            weighted_load *= (1 / self.t) / sum((1 / self.t))
        if sum(self.c + self.r) > 0:
            weighted_load *= (self.c + self.r) / sum(self.c + self.r)

        x0 = self.a + weighted_load / sum(weighted_load) * (
                self.S - sum(self.a))
        lb_bound = self.a  # lb <= A.dot(x) <= ub
        ub_bound = self.S - dot((ones((self.J, self.J)) - eye(self.J)), self.a)
        bounds = optimize.Bounds(lb_bound, ub_bound)

        A_cons = array([1] * self.J)
        lb_cons = self.S  # Equal bounds represent equality constraint
        ub_cons = self.S
        lin_cons = optimize.LinearConstraint(A_cons, lb_cons, ub_cons)

        s_star = optimize.minimize(self.server_allocation_cost, x0,
                                   bounds=bounds, constraints=lin_cons).x
        return s_star

    @staticmethod
    def def_sizes(dim):
        """Docstring."""
        sizes = np.zeros(len(dim), dtype=np.int64)
        sizes[-1] = 1
        for i in range(len(dim) - 2, -1, -1):
            sizes[i] = sizes[i + 1] * dim[i + 1]
        return sizes

    def init_Pi(self):
        """
        Take the longest waiting queue into service (or last queue if tied).
        Take arrivals directly into service.
        """
        Pi = self.NOT_EVALUATED * np.ones(self.dim_i, dtype=int)
        for s in self.s_states_v:
            states = np.append([slice(None)] * (1 + self.J), s)
            if np.sum(s) == self.S:
                Pi[tuple(states)] = self.SERVERS_FULL
                continue
            for i in arange(self.J):
                states_ = states.copy()
                for x in arange(1, self.D + 1):
                    states_[1 + i] = x  # x_i = x
                    for j in arange(self.J):
                        if j != i:
                            states_[1 + j] = slice(0, x + 1)  # 0 <= x_j <= x_i
                    Pi[tuple(states_)] = i + 1
                states_ = states.copy()
                states_[0] = i
                states_[1 + i] = 0
                Pi[tuple(states)] = i + 1  # Admit arrival (of i)
            states = np.concatenate(([self.J], [0] * self.J, s),
                                    axis=0)  # x_i = 0 All i
            Pi[tuple(states)] = self.NONE_WAITING
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
            self.test_loop(memory, self.size_i, self.sizes_i, self.s_states,
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
