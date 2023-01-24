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
from numpy import array, arange, zeros
from itertools import product
from sys import exit, getsizeof
from timeit import default_timer
from numba import njit

class TimeConstraintEDs():
    """Environment of a Time Constraint Emergency Department."""

    S_MIN = 2  # Servers
    S_MAX = 20
    mu_MIN = 0.1  # Service Rate
    mu_MAX = 1
    load_MIN = 0.4  # System load
    load_MAX = 1
    imbalance_MIN = 1  # Imbalance
    imbalance_MAX = 5
    TARGET = [1/3]  # [1/3] or np.linspace(1/3, 10/3, num=10)  # Target

    def __init__(self, **kwargs):  # **kwargs: Keyworded arguments
        """Create all variables describing the environment."""
        self.J = kwargs.get('J')
        self.S = kwargs.get('S', np.random.randint(self.S_MIN,
                                                   self.S_MAX+1))  # [a, b)
        self.mu = kwargs.get('mu', np.random.uniform(self.mu_MIN,
                                                     self.mu_MAX,
                                                     self.J))
        if 'lambda_' in kwargs:
            self.lambda_ = kwargs.get('lambda_')
            self.rho = sum(self.lambda_) / (sum(self.mu) * self.S)
        else:  # Determine arrival rate based on desired load.
            self.rho = kwargs.get('rho',
                                  np.random.uniform(self.load_MIN,
                                                    self.load_MAX))
            weight = np.random.uniform(self.imbalance_MIN,
                                       self.imbalance_MAX, self.J)
            self.lambda_ = self.mu * (self.S*self.rho*weight/sum(weight))

        self.t = kwargs.get('t', np.random.choice(self.TARGET, self.J))
        self.c = kwargs.get('c', [1]*self.J)

        self.gamma = kwargs.get('gamma')
        self.D = kwargs.get('D')
        self.P = kwargs.get('P')
        self.e = kwargs.get('e')

        self.max_iter = kwargs.get('max_iter')
        self.trace = kwargs.get('trace')
        self.print_modulo = kwargs.get('print_modulo')

        self.alpha = self.t*self.gamma
        self.P_xy = self.trans_prob()
        self.tau = self.S*max(self.mu) + \
            sum(np.maximum(self.lambda_, self.gamma))
        self.dim = (self.D+1, self.S+1)*self.J
        self.size = np.prod(self.dim)
        self.sizes = self.def_sizes()

        self.S_states = array(list(product(arange(self.S+1), repeat=self.J)))
        self.S_states = self.S_states[np.sum(self.S_states, axis=1) <= self.S]
        self.x_states = array(list(product(arange(self.D+1), repeat=self.J)))

        self.feasibility(kwargs.get('time_check'))
        if self.trace:
            print("J =", self.J, ", D =", self.D, ", s =", self.s, '\n',
                  "lambda:", round(self.lambda_, 4), '\n',
                  "mu:", round(self.mu, 4), '\n',
                  "a:", round(self.lambda_/self.mu, 4), '\n',
                  "rho:", round(self.rho, 4), '\n',
                  "Target:", round(self.t, 4))
        assert self.rho < 1, "rho < 1 does not hold"

    def trans_prob(self):
        """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
        P_xy = np.zeros((self.J, self.D+1, self.D+1))
        gamma = self.gamma
        A = np.indices((self.D+1, self.D+1))  # x=A[0], y=A[1]

        for i in range(self.J):  # For every class
            lambda_ = self.lambda_[i]
            mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
            for i in range(self.J):
                P_xy[i, 1:, 1:][mask_tril] = (gamma / (lambda_ + gamma)) ** \
                    (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
                    lambda_ / (lambda_ + gamma)
                P_xy[i, 1:, 0] = (gamma / (lambda_ + gamma)) ** A[0, 1:, 0]
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

    @njit
    def test_loop(self, memory):
        """Docstring."""
        memory = memory.reshape(self.size); J = self.J
        for s in self.S_states:
            for x in self.x_states:
                state = np.sum(x*self.sizes[0:J] + s*self.sizes[J:J*2])
                memory[state] = np.random.rand()

    def feasibility(self, time_check):
        """Feasability Check matrix size and rho<1."""
        memory = zeros(self.dim)
        if self.trace:
            print("GB: ", round(getsizeof(memory)/10**9, 4))
        if time_check:
            self.timer(True, 'Feasibility', True)
            time = self.test_loop(memory)  # Numba
            time = self.timer(False, 'Feasibility', True)
            if(time > 60):  # in seconds
                exit("Looping matrix takes more than 60 seconds.")

    @njit
    def W(self, V, W, Pi):
        """W."""
        J = self.J
        sizes = self.sizes
        V = V.reshape(self.size)
        W = W.reshape(self.size)
        Pi = Pi.reshape(self.size)
        for s in self.S_states:
            for x in self.x_states:
                state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
                i = Pi[state]
                if i == J + 1:
                    W[state] = V[state] + \
                        (self.P if np.any(x == self.D) else 0)
                elif i >= 0:  # If someone of class i waiting
                    W[state] = self.c[i-1] if x[i-1] > self.alpha[i-1] else 0
                    for y in arange(x[i-1] + 1):
                        next_x = x.copy()
                        next_x[i-1] = y
                        next_s = s.copy()
                        next_s[i-1] += 1
                        next_state = np.sum(next_x*sizes[0:J] + \
                                            next_s*sizes[J:J*2])
                        W[state] += self.P_xy[i-1, x[i-1], y] * V[next_state]
        return V.reshape(self.dim), W.reshape(self.dim), Pi.reshape(self.dim)

    def V(self, V, W):
        """V_t."""
        all_states = [slice(None)]*(self.J*2)
        V_t = self.tau * V
        for i in arange(self.J):
            # x_i = 0
            states = all_states.copy()
            next_states = all_states.copy()
            states[i] = 0
            next_states[i] = 1
            V_t[tuple(states)] += self.lambda_[i] * (W[tuple(next_states)] -
                                                     V[tuple(states)])
            # 0 < x_i < D
            states = all_states.copy()
            next_states = all_states.copy()
            states[i] = slice(1, self.D)
            next_states[i] = slice(2, self.D+1)
            V_t[tuple(states)] += self.gamma * (W[tuple(next_states)] -
                                                V[tuple(states)])
            # x_i = D
            states = all_states.copy()
            states[i] = self.D
            V_t[tuple(states)] += self.gamma * (W[tuple(states)] -
                                                V[tuple(states)])
            # s_i
            for s_i in arange(1, self.S+1):
                states = all_states.copy()
                next_states = all_states.copy()
                states[self.J+i] = s_i
                next_states[self.J+i] = s_i - 1
                V_t[tuple(states)] += s_i * self.mu[i] * \
                    (W[tuple(next_states)] - V[tuple(states)])
        return V_t/self.tau

    def convergence(env, V_t, V, counter):
        """Convergence check of valid states only."""
        delta_max = V_t[tuple([0]*(env.J*2))] - V[tuple([0]*(env.J*2))]
        delta_min = delta_max.copy()
        for s in env.S_states:
            states = [slice(None)]*(env.J*2)
            states[slice(env.J,env.J*2)] = s
            diff = np.abs(V_t[tuple(states)] - V[tuple(states)])
            delta_max = np.max([np.max(diff), delta_max])
            delta_min = np.min([np.min(diff), delta_min])
            if delta_max - delta_min > env.e:
                break
        converged = delta_max - delta_min < env.e
        max_iterations = counter > env.max_iter
        if (converged & env.trace) | max_iterations:
            print("iter: ", counter,
                  ", delta: ", round(delta_max - delta_min, 2),
                  ', D_min', round(delta_min, 2),
                  ', D_max', round(delta_max, 2),
                  ", g: ", round((delta_max + delta_min)/(2 * env.tau), 2))
        if max_iterations:
            exit('Max iterations reached')
        return converged | max_iterations

    @njit
    def policy_improvement(self, V, Pi):
        """Determine best action/policy per state by one-step lookahead."""
        J = self.J
        sizes = self.sizes
        V = V.reshape(self.size)
        Pi = Pi.reshape(self.size)
        unstable = False

        for s in self.S_states:
            for x in self.x_states:
                state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
                pi = Pi[state].copy()
                w = V[state] + (self.P if np.any(x == self.D) else 0)
                Pi[state] = J + 1
                if np.sum(s) < self.S:
                    for i in arange(J):
                        if(x[i] > 0):  # If someone of class i waiting
                            value = self.c[i] if x[i] > self.alpha[i] else 0
                            for y in arange(x[i] + 1):
                                next_x = x.copy()
                                next_x[i] = y
                                next_s = s.copy()
                                next_s[i] += 1
                                next_state = np.sum(next_x*sizes[0:J] + \
                                                    next_s*sizes[J:J*2])
                                value += self.P_xy[i, x[i], y] * V[next_state]
                            Pi[state] = i + 1 if value < w else Pi[state]
                            w = array([value, w]).min()
                if pi != Pi[state]:
                    unstable = True
        return V.reshape(self.dim), Pi.reshape(self.dim), unstable
