# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 16:21:36 2020

@author: Jeroen
"""

s_MIN = 2  # Servers
s_MAX = 20
mu_MIN = 0.1  # Service Rate
mu_MAX = 1
load_MIN = 0.4  # System load
load_MAX = 1
imbalance_MIN = 1  # Imbalance
imbalance_MAX = 5
TARGET = [1/3]  # [1/3] or np.linspace(1/3, 10/3, num=10)  # Target
INVALID = -np.inf  # Makes sure that x + inf --> inf.

def __init__(self, J, gamma, epsilon, D, penalty, trace, timeout,
             printed_states, **kwargs):
    """Create all variables describing the environment."""
    # **kwargs: Keyworded arguments
    # Classes
    self.J = J
    # Servers
    self.s = kwargs.get('s', np.random.randint(self.s_MIN,
                                               self.s_MAX+1))  # [a, b)
    # Service rate list
    self.mu = kwargs.get('mu', np.random.uniform(self.mu_MIN,
                                                 self.mu_MAX,
                                                 self.J))
    # Total system load
    rho = kwargs.get('rho', np.random.uniform(self.load_MIN,
                                              self.load_MAX))
    # Arrival rate list
    self.lambda_ = kwargs.get('lambda_', self.init_lambda(rho))
    # target time list
    self.t = kwargs.get('t', np.random.choice(self.TARGET, self.J))
    self.weight = kwargs.get('weight', [1]*self.J)
    self.gamma = gamma  # Discritization parameter
    self.alpha = self.t*self.gamma
    self.epsilon = epsilon  # Small number
    self.penalty = penalty  # Penalty never starting service
    self.D = D  # Cap on time
    self.P_xy = self.trans_prob()
    self.dim = (self.D+1, self.s+1)*self.J # Include 0 states
    self.tau = self.s*max(self.mu) + \
        sum(np.maximum(self.lambda_, self.gamma))
    self.timeout = timeout
    print("J=", J, ", D=", D, ", s=", self.s)
    print("lambda:", np.round(self.lambda_, 4))
    print("mu:", np.round(self.mu, 4))
    print("a:", np.round(self.lambda_/self.mu, 4))
    print("weight:", np.round(self.weight, 4))
    print("Target:", np.round(self.t, 4))
    self.trace = trace
    self.print_modulo = int(((D+1)*(self.s+1))**J/printed_states)
    self.combinations()

def init_lambda(self, rho):
    """Determine arrival rate based on desired load."""
    weight = np.random.uniform(self.imbalance_MIN,
                               self.imbalance_MAX, self.J)
    return self.mu * (self.s*rho*weight/sum(weight))  # mu * (queue load)