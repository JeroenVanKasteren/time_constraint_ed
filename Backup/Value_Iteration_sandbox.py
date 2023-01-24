"""
Interactively coding Value Iteration.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
"""

import numpy as np
from numpy import array, arange
from numba import njit
from numba.core import types
from numba.typed import Dict
from itertools import product

np.set_printoptions(precision=3)
MAX_ITER = 10

env = Dict.empty(key_type=types.unicode_type,
                 value_type=types.float64)
env1 = Dict.empty(key_type=types.unicode_type,
                   value_type=types.float64[:])
env2 = Dict.empty(key_type=types.unicode_type,
                    value_type=types.float64[:,:])
env3 = Dict.empty(key_type=types.unicode_type,
                     value_type=types.float64[:,:,:])
env['J'] = 2
env['s'] = 3
env['D'] = 4

env1['mu'] = array([0.6, 0.6])
env1['lambda_'] = array([3/4, 3/4])
env1['t'] = array([0., 0.])
env1['weight'] = array([1., 1.])

env['gamma'] = 1
env['P'] = 1e2
env['e'] = 1e-5

env['trace'] = 0  # levels: 0 nothing, 1 results, 2 iterations, 3 all
env['no_states_printed'] = 5
env['iter'] = 0
env['max_iter'] = 10

env['tau'] = env['s']*np.max(env1['mu']) + \
    np.sum(np.maximum(env1['lambda_'], env['gamma']))
env1['dim'] = np.repeat([env['D']+1, env['s']+1], env['J'])
env['size'] = np.prod(env1['dim'])

env1['V'] = np.zeros(int(env['size']), dtype=float)  # V_{t-1}
env1['V_t'] = np.zeros(int(env['size']), dtype=float)  # V
env1['W'] = np.zeros(int(env['size']), dtype=float)
env['g'] = 0
env['print_modulo'] = int(((env['D']+1) * (env['s']+1)) ** env['J'] /
                        env['no_states_printed'])

@njit
def size_per_dim(env1):
    """Return array with size per dimension."""
    sizes = np.zeros(len(env1['dim']))
    sizes[-1] = 1
    for i in range(len(env1['dim']) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * env1['dim'][i + 1]
    env1['sizes'] = sizes


def combinations(env, env2):
    """Calculate all valid server and waiting time states."""
    s_states = array(list(product(arange(env['s']+1), repeat=int(env['J']))))
    env2['s_states'] = s_states[np.sum(s_states, axis=1) <= env['s']]
    env2['x_states'] = array(list(product(arange(env['D']+1),
                                         repeat=int(env['J']))))


@njit
def valid_states(env, env1, env2):
    """Get list of all valid states."""
    J = int(env['J'])
    no_x_states = len(env2['x_states'])
    states = np.zeros(no_x_states*len(env2['s_states']))
    i = 0
    for s_state in env2['s_states']:
        states[i:(i+no_x_states)] = \
            np.sum(env2['x_states']*env1['sizes'][0:J] +
                   s_state*env1['sizes'][J:J*2], axis=1)
        i = i + no_x_states
    env1['states'] = states


def trans_prob(env, env1, env3):
    """P_xy(i, x, y) transition prob. for class i to jump from x to y."""
    P_xy = np.zeros((int(env['J']), int(env['D']+1), int(env['D']+1)))
    gamma = env['gamma']
    A = np.indices((int(env['D']+1), int(env['D']+1)))  # x=A[0], y=A[1]

    for i in arange(int(env['J'])):  # For every class
        lambda_ = env1['lambda_'][i]
        mask_tril = A[0, 1:, 1:] >= A[1, 1:, 1:]
        P_xy[i, 1:, 1:][mask_tril] = (gamma / (lambda_ + gamma)) ** \
            (A[0, 1:, 1:][mask_tril] - A[1, 1:, 1:][mask_tril]) * \
            lambda_ / (lambda_ + gamma)
        P_xy[i, 1:, 0] = (gamma / (lambda_ + gamma)) ** A[0, 1:, 0]
    env3['P_xy'] = P_xy


size_per_dim(env1)
combinations(env, env2)
valid_states(env, env1, env2)
trans_prob(env, env1, env3)


# def value_iteration(env, env1, env2, env3):
while True:  # Update each state.
    one_step_lookahead(env, env1, env2, env3)

    if stopping_condition(env, env1):
        break

    # Save and Rescale the value function
    env['V'] = env['V_t'].copy() - env['V_t'][np.repeat(0, 2*int(env['J']))]
    env['iter'] = env['iter'] + 1
env.timer(False, self.name)


def c(self, x, i):
    """Input x must be a numpy array."""
    return np.where(x > self.alpha[i], self.weight[i], 0)


def c_D(self, x):
    """Penalty for not taking into service."""
    return self.penalty if any(np.array(x) == self.D) else 0


def one_step_lookahead(env, env1, env2, env3):
    """One-step lookahead with V = V_{t-1} and V_t = V_{t}."""
    for s_state in self.s_states:
    all_states = [slice(None)]*(self.J*2)
    V_t[:] = self.tau * V[tuple(all_states)]
    for i in range(self.J):
        # x_i = 0
        states = all_states.copy(); next_states = all_states.copy()
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
    V = V_t / self.tau


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
    """
    Return value given policy.
    """
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


def stopping_condition(env, env1):
    """Convergence or max_iter."""
    diff = np.abs(env1['V_t'][env1['states']] - env1['V'][env1['states']])
    delta_max = max(diff)
    delta_min = min(diff)

    if trace > 1: print("Delta: ", np.round(delta_max - delta_min, 2)
    if delta_max - delta_min < env['e']:
        if trace > 0: print("g: ", (delta_max + delta_min)/2 * tau, "\n",
                            "Iterations: ", env['iter'])
        return True
    elif counter > max_iter:
        if trace > 0: print('Value Iteration reached max_iter', env['iter'])
        return True
    return False

# def timer(self, start_boolean, name):
#     """Docstring."""
#     if not self.trace:
#         pass
#     elif start_boolean:
#         print('Starting ' + name + '.')
#         self.start = timeit.default_timer()
#     else:
#         time = timeit.default_timer() - self.start
#         print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))
#         print('Finished ' + name + '.')
