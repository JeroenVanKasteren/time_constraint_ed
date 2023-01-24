"""
MDP environment class.

All elements of 'self':

Input variables, Necessairy
'J' classes, int, J > 1
'gamma' time discritization parameter, float, gamma > 0
'D' waiting time transitions cap, int, assumed D > 1
'P' penalty never starting service, flaot, P >= 0
'e' epsilon (small number), float, used for convergence check, 0 < e << 1

Input Programming variables, Necessairy
'max_iter' of Value or Policy Iteration (VI and PI), int
'trace' print time and convergence? boolean
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
from numpy import array, arange, zeros, round
from itertools import product
from sys import exit, getsizeof
from timeit import default_timer
from numba import njit


class TimeConstraintEDs():
    S_MIN = 2  # Servers
    S_MAX = 6
    mu_MIN = 0.1  # Service Rate
    mu_MAX = 1
    load_MIN = 0.4  # System load
    load_MAX = 1
    imbalance_MIN = 1  # Imbalance
    imbalance_MAX = 5
    TARGET = [1]  # [x] or np.linspace(x, 10*x, num=10)  # Target

    def __init__(self, **kwargs):  # **kwargs: Keyworded arguments
        """Create all variables describing the environment."""
        self.J = kwargs.get('J')
        self.S = kwargs.get('S', np.random.randint(self.S_MIN, self.S_MAX+1))  # [a, b)
        self.mu = kwargs.get('mu', np.random.uniform(self.mu_MIN, self.mu_MAX, self.J))
        if 'lmbda' in kwargs:
            self.lmbda = kwargs.get('lmbda')
            self.Rho = array(sum(self.lmbda / self.mu) / self.S)
        else:  # Determine arrival rate based on desired load.
            self.Rho = kwargs.get('Rho', np.random.uniform(self.load_MIN, self.load_MAX))
            weight = np.random.uniform(self.imbalance_MIN, self.imbalance_MAX, self.J)
            self.lmbda = self.mu * (self.S*self.rho*weight/sum(weight))
        self.t = kwargs.get('t', np.random.choice(self.TARGET, self.J))
        self.gamma = kwargs.get('gamma')
        if any((self.t % (1/self.gamma) != 0) | (self.t < 1/self.gamma)):
            self.t = np.floor(self.t*self.gamma) / self.gamma
            print("Rounded t down to nearest multiple of 1/gamma.")
        self.c = kwargs.get('c', array([1]*self.J))
        self.r = kwargs.get('r', array([1]*self.J))
        self.P = kwargs.get('P', max(self.c + self.r)*10)
        self.D = kwargs.get('D')
        self.e = kwargs.get('e', 1e-5)
        
        self.a = self.lmbda/self.mu
        self.P_xy = self.trans_prob()
        self.tau = self.S * max(self.mu) + sum(np.maximum(self.lmbda, self.gamma))
        self.dim = tuple(np.repeat([self.D + 1, self.S + 1], self.J))
        self.sizes = self.def_sizes(self.dim)
        self.size = np.prod(self.dim)
        self.dim_i = tuple(np.append(self.J + 1, np.repeat([self.D + 1, self.S + 1], self.J)))
        self.sizes_i = self.def_sizes(self.dim_i)
        self.size_i = np.prod(self.dim_i)

        self.max_iter = kwargs.get('max_iter', np.Inf)
        self.trace = kwargs.get('trace', False)
        self.print_modulo = kwargs.get('print_modulo', 1)

        self.s_states = array(list(product(arange(self.S + 1), repeat=self.J)))
        self.s_states_v = self.s_states[np.sum(self.s_states, axis=1) <= self.S]  # Valid states
        self.s_states = self.s_states_[np.sum(self.s_states_, axis=1) < self.S]  # Action states
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
              "rho:", round(self.rho, 4))
        assert self.Rho < 1, "rho < 1 does not hold"

    def trans_prob(self):
        """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
        P_xy = np.zeros((self.J, self.D+1, self.D+1))
        gamma = self.gamma
        A = np.indices((self.D+1, self.D+1))  # x=A[0], y=A[1]
        mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]    
        for i in range(self.J):  # For every class
            lmbda = self.lmbda[i]
            P_xy[i, 1:, 1:][mask_tril] = (gamma / (lmbda + gamma)) ** \
                (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
                lmbda / (lmbda + gamma)
            P_xy[i, 1:, 0] = (gamma / (lmbda + gamma)) ** A[0, 1:, 0]
        P_xy[:, 0, 0] = 1
        return P_xy

    def def_sizes(self):
        """Docstring."""
        sizes = np.zeros(len(self.dim), dtype=np.int64)
        sizes[-1] = 1
        for i in range(len(self.dim) - 2, -1, -1):
            sizes[i] = sizes[i + 1] * self.dim[i + 1]
        return sizes

    def timer(self, start_boolean, name, trace):
        """Only if trace=TRUE, start timer if start=true, else print time."""
        if not trace:
            pass
        elif start_boolean:
            print('Starting ' + name + '.')
            self.start = default_timer()
        else:
            time = default_timer() - self.start
            print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))
            print('Finished ' + name + '.')
            return time

    @staticmethod
    @njit
    def test_loop(memory, size, sizes, S_states, x_states, J):
        """Docstring."""
        memory = memory.reshape(size)
        for s in S_states:
            for x in x_states:
                state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
                memory[state] = np.random.rand()

    def feasibility(self, time_check):
        """Feasability Check matrix size."""
        memory = zeros(self.dim)
        name = 'Running time feasibility'
        if self.trace:
            print("Size of V: ", round(getsizeof(memory)/10**9, 4), "GB.")
        if time_check:
            self.timer(True, name, True)
            time = self.test_loop(memory, self.size, self.sizes,
                                  self.S_states, self.x_states, self.J)  # Numba
            time = self.timer(False, name, True)
            if(time > 60):  # in seconds
                exit("Looping matrix takes more than 60 seconds.")

