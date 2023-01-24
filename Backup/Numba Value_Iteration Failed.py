"""
Created on 7-10-2020.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)

Description: The class containing learner functies to take actions and learn
from experience. (The learner is also known as the agent or actor.)

The learner functions implemented are:
    - Value Iteration
    - Policy Iteration (PI)
    - One Step Policy Iteration
"""

import numpy as np
from numpy import array, arange
from numba import jitclass
from numba.core import types
from itertools import product

spec = [('name', types.unicode_type),
        ('V', types.float32[:]),
        ('V_t', types.float32[:]),
        ('W', types.float32[:]),
        ('states', types.float32[:]),
        ('g', types.float32),
        ('max_iter', types.int32)]

spec = [('J', types.float32[:]),
        ('s', types.float32[:]),
        ('D', types.float32[:]),
        ('mu', types.float32[:]),
        ('lambda', types.float32[:]),
        ('t', types.float32[:]),
        ('w', types.float32[:]),
        ('gamma', types.float32[:]),
        ('P', types.float32[:]),
        ('e', types.float32[:]),
        ('trace', types.float32[:]),
        ('no_states_printed', types.float32[:]),
        ('max_iter', types.float32[:]),
        ('tau', types.float32[:]),
        ('dim', types.float32[:]),
        ('size', types.float32[:]),
        ('name', types.unicode_type),
        ('V', types.float32[:]),
        ('V_t', types.float32[:]),
        ('W', types.float32[:]),
        ('states', types.float32[:]),
        ('g', types.float32),
        ('max_iter', types.int32)]


@jitclass(spec)
class ValueIteration():
    """Value Iteration in Numba."""

    def __init__(self, env):
        self.name = 'Value Iteration'
        self.V = np.zeros(env.dim, )  # V_{t-1}
        self.V_t = self.V.copy()  # V_{t}
        self.W = self.V.copy()  # V_{t}
        self.g = 0

    def value_iteration(self):
        """Docstring."""
        counter = 0
        while True:  # Update each state.
            one_step_lookahead(self.V_t, self.V, self)

            if (env.convergence(self.V_t, self.V, counter) or
                    counter == self.max_iter):
                break  # Stopping condition

            # Save and Rescale the value function
            self.V = self.V_t.copy() - self.V_t[(0, 0)*self.J]
            counter += 1
        env.timer(False, self.name)
