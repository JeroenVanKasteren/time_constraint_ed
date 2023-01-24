"""
MDP environment class.

Input Constants
'J' classes, int, J > 1
's' servers, float, s > 0
total system load 'rho', float, 0 < rho < 1, rho = lambda / (s*mu)

service rate 'mu', array, mu > 0
arrival rate 'lambda', array, lambda > 0
target time 't', array, t >= 0
(Penlaty) 'weight', np.array

time discritization parameter 'gamma', float, >0
'D' waiting time transitions cap, assumed >1
'penalty', flaot,  for never starting service
'epsilon', float, small constant for convergence check

Input Programming Constants
print time and convergence? 'trace', boolean
'max_iter' of Value or Policy Iteration (VI and PI), int
'no_states_printed' No. times the current state of iterating is printed

Dependent variables
'alpha' No. gamma transitions allowed to wait
'P_xy'(i, x, y) transition prob. for class i to jump from x to y
'tau' uniformization constant
'dim' (D+1, s+1), +1 to include 0 state


@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
from numpy import array
from numpy import arange
# from numpy import zeros
# from numpy import round
# from numpy import tile
from numpy import inf
import itertools
from scipy import optimize
from scipy.integrate import quad_vec
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma
from sys import exit
import timeit


class TimeConstraintEDs():
    """Environment of a Time Constraint Emergency Department."""

    INVALID = -inf  # Makes sure that x + inf --> inf.

    def __init__(self, **kwargs):  # **kwargs: Keyworded arguments
        """Create all variables describing the environment."""
        self.J = array(kwargs.get('J'))
        self.s = array(kwargs.get('s'))

        self.mu = array(kwargs.get('mu'))
        self.lambda_ = array(kwargs.get('lambda_'))
        self.t = array(kwargs.get('t'))
        self.weight = array(kwargs.get('weight'))

        self.gamma = kwargs.get('gamma')
        self.D = kwargs.get('D')
        self.penalty = kwargs.get('penalty')
        self.epsilon = kwargs.get('epsilon')

        self.max_iter = kwargs.get('max_iter')
        self.trace = kwargs.get('trace')
        no_states_printed = kwargs.get('no_states_printed')

        self.alpha = self.t*self.gamma
        self.tau = self.s*max(self.mu) + \
            np.sum(np.maximum(self.lambda_, self.gamma))
        self.dim = (self.D+1, self.s+1)*self.J

        self.print_modulo = int(((self.D+1) * (self.s+1)) ** self.J /
                                no_states_printed)
        self.combinations()

    def combinations(self):
        """Calculate all valid server and waiting time combinations."""
        s_states = itertools.product(arange(self.s+1), repeat=self.J)
        s_states = array(list(s_states))
        self.s_states = s_states[np.sum(s_states, axis=1) <= self.s]
        self.x_states = array(list(itertools.product(arange(self.D+1),
                                                     repeat=self.J)))

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

    def c(self, x, i):
        """Input x must be a numpy array."""
        return np.where(x > self.alpha[i], self.weight[i], 0)

    def penalty(self, x):
        """Penalty for not taking into service."""
        return self.penalty if any(np.array(x) == self.D) else 0

    def one_step_lookahead(self, V_t, V, W):
        """One-step lookahead with V = V_{t-1} and V_t = V_{t}."""
        self.calculate_W(V_t, W)
        all_states = [slice(None)]*(self.J*2)
        V_t = self.tau * V
        # V_t[:] = self.tau * V[tuple(all_states)]
        for s_state in self.s_states:
            # x_i = 0
            states = all_states.copy()
            next_states = all_states.copy()
            states[i*2] = 0
            next_states[i*2] = 1
            V_t[tuple(states)] += self.lambda_[i] * W[tuple(next_states)] -\
                self.lambda_[i] * V[tuple(states)]
            # 0 < x_i < D
            states = all_states.copy(); next_states = all_states.copy()
            states[i*2] = slice(1, self.D)
            next_states[i*2] = slice(2, self.D+1)
            V_t[tuple(states)] += self.gamma * W[tuple(next_states)] -\
                self.gamma * V[tuple(states)]
            # x_i = D
            states = all_states.copy(); next_states = all_states.copy()
            states[i*2] = self.D
            next_states[i*2] = self.D
            V_t[tuple(states)] += self.gamma * W[tuple(next_states)] -\
                self.gamma * V[tuple(states)]
            # s_i
            for s_i in range(1, self.s+1):
                states = all_states.copy(); next_states = all_states.copy()
                states[i*2+1] = s_i
                next_states[i*2+1] = s_i - 1
                V_t[tuple(states)] += s_i * self.mu[i] * \
                    W[tuple(next_states)] - \
                        s_i * self.mu[i] * V[tuple(states)]
        V_t[:] = V_t / self.tau

    def calculate_W(self, V, W):
        """Docstring."""
        for s_state in self.s_states:
            for d_state in self.x_states:
                state = tuple(np.array([d_state, s_state]).T.reshape(-1))
                if sum(s_state) == self.s:
                    W[state] = V[state] + self.penalty(d_state)
                    continue
                for i in range(self.J):
                    w = V[state] + self.penalty(d_state)
                    for i in range(self.J):
                        tmp = c(x)
                        for x in self.D_1_states:
                            x = list(x)
                            x.insert(i, slice(None))
                            next_state = np.array([x, s_state]).T.reshape(-1)
                            next_state[i*2] = slice(self.D, None, -1)  # x_i-(x_i-y)
                            next_state[i*2+1] += 1  # s_i + 1
                            w[tuple([i]+x)] = self.c(np.arange(self.D+1), i) +\
                                np.sum(self.P_xy[i, slice(None), slice(None)] *\
                                       V[tuple(next_state)], axis=1)
                            W[states] = np.min(
                                np.concatenate((w, [V[states] + penalty_m]),0),0)
        return W

    def W(self, multi_state, V):
        """Return value."""
        states = np.array(multi_state).reshape(self.J, 2)
        if(sum(states[:,1]) == self.s):
            return V[multi_state]
        w = np.full(self.J+1, np.inf)
        for i in range(self.J):
            x_i = multi_state[i*2]
            if(x_i > 0):  # If someone of class i waiting
                next_state = list(multi_state)
                next_state[i*2] = list(range(x_i, -1, -1))
                next_state[i*2+1] += 1  # s
                w[i] = self.c(x_i, i) + \
                    np.sum(self.P_xy[i, x_i, range(x_i+1)] * \
                           V[tuple(next_state)])
            else:  # Else no one of class i to take into service
                w[i] = np.inf  # Make sure w[i] > self.V[multi_state]
        w[self.J] = V[multi_state]  # do nothing
        return np.min(w)

    def W_i(self, multi_state, i, policy, V):
        """Return value given policy."""
        states = np.array(multi_state).reshape(self.J, 2)
        if((policy[multi_state] == self.J) |
           (sum(states[:,1]) == self.s) | (sum(states[:,0]) == 0)):
            policy[multi_state] == self.J
            return V[multi_state]
        w = 0
        x_i = multi_state[i*2]
        if(x_i > 0):  # If someone of class i waiting
            next_state = list(multi_state)
            next_state[i*2] = list(range(x_i, -1, -1))
            next_state[i*2+1] += 1  # s
            w = self.c(x_i, i) + \
                np.sum(self.P_xy[i, x_i, range(x_i+1)] * \
                       V[tuple(next_state)])
        else:  # Else no one of class i to take into service
            w = V[multi_state]  # action: do nothing
        return w

    def invalid_entries(self):
        """Mark all invalid entries of table."""
        matrix = np.zeros((self.D+1, self.s+1)*self.J)
        s_states = itertools.product(range(self.s+1), repeat=self.J)
        s_states = np.array(list(s_states))
        s_states = s_states[np.sum(s_states, axis=1) > self.s]
        for s_state in s_states:
            multi_state = np.array(
                [[slice(None)]*self.J, s_state]).T.reshape(-1)
            matrix[tuple(multi_state)] = self.INVALID
        return matrix

    def timer(self, start_boolean, name):
        """Docstring."""
        if not self.trace:
            pass
        elif start_boolean:
            print('Starting ' + name + '.')
            self.start = timeit.default_timer()
        else:
            time = timeit.default_timer() - self.start
            print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))
            print('Finished ' + name + '.')

    def convergence(self, V_t, V, counter):
        """Docstring."""
        print("V_t", V_t)  # TODO
        diff = np.abs(V_t[np.isfinite(V_t)] - V[np.isfinite(V)])
        self.delta_max = max(diff)
        self.delta_min = min(diff)
        print("Delta", self.delta_max - self.delta_min)
        converged = self.delta_max - self.delta_min < self.epsilon
        if(converged):
            if self.trace: print("delta: ", self.delta_max - self.delta_min,
                                 "g: ", (self.delta_max + self.delta_min)/2*self.tau,
                                 ", Iterations: ", counter)
            return converged
        elif counter > self.max_iter:
            if self.trace:
                print('Value Iteration reached max_iter')
                print("delta:",np.round(self.delta_max-self.delta_min,2),
                      ", Iterations: ", counter)
            return True

    def V_to_memory(self, _s, i, method, s_star):  # _s Extra
        """
        Calculate V for a single queue for all x = -s, ..., 0, ..., D.

        Input
        _s: is the number of servers available for the queue
        i: the class
        approx_method: indicates the approximation method (see article)

        Only call this function once per class for effeciency.
        """
        lambda_ = self.lambda_[i]; mu = self.mu[i]; t = self.t[i]
        weight = self.weight[i]; gamma = self.gamma
        s_n = int(_s)
        a = lambda_ / mu; rho = a / _s

        # Extra, calculate once vectorized, use pi_0[i], g[i], ...
        pi_0 = self.get_pi_0(_s, rho, i=i)
        tail_prob = self.get_tail_prob(_s, rho, pi_0, i=i)
        # Scale to get average reward
        g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

        V = np.zeros(s_n+1+self.D)  # V(-s) = 0, reference state
        if((method != 1) & (_s == s_star[i])):  # Extra
            x = np.arange(-s_n+1, 1)
            V_x_le_0 = lambda y: (1 - (y/a)**(x+_s)) / (1 - y/a) * np.exp(a-y)
            V[x+s_n] = g/lambda_ * quad_vec(V_x_le_0, a, np.inf)[0]

        x = np.array(range(1, self.D+1))
        tmp_frac = (_s*mu + gamma) / (lambda_ + gamma)
        V[x+s_n] = V[s_n] + g / (gamma*_s*mu*(1 - rho)**2) * \
            (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x -
             (lambda_ + gamma))
        _sum = np.exp(a) / a**(_s-1) * gamma_fun(_s) * reg_up_inc_gamma(_s, a)
        V[x+s_n] += g / (lambda_*gamma*(rho-1)) * \
            (lambda_ - gamma - gamma/rho * _sum) * \
            (-rho + (lambda_ + gamma) / (_s*mu + gamma) * tmp_frac**x)
        # -1_{x > t*gamma}[...]
        alpha = np.floor(t*gamma+1).astype(int)
        x = np.array(range(alpha, self.D+1))
        V[x+s_n] -= weight/(gamma * (1 - rho)**2) * \
            ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) +
             (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
        return V

