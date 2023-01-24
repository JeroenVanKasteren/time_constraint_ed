"""
Created on 19-3-2020.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)

Description: The class with the MDP environment.

mu, lambda and similar vectors are assumed to be numpy arrays.
"""

import numpy as np
from scipy import optimize
from scipy.integrate import quad_vec
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma

class Environment():
    """The environment interface for a Markov Decision Process (MDP)."""

    def __init__(self):
        pass

    def reset(self):
        """Docstring."""
        return []

    def next_state(self, state, action):
        """Docstring."""
        done = False
        ns = state
        reward = self.reward(state, action, ns)
        return (ns, reward, done)

    def get_actions(self):
        """Docstring."""
        return []

    def reward(self, state, action, next_state):
        """Docstring."""
        return -1


class TimeConstraintEDs(Environment):
    """Environment of a Time Constraint Emergency Department."""

    s_MIN = 2  # Servers
    s_MAX = 20
    mu_MIN = 0.1  # Service Rate
    mu_MAX = 1
    load_MIN = 0.4  # System load
    load_MAX = 1
    imbalance_MIN = 1  # Imbalance
    imbalance_MAX = 5
    TARGET = [1/3]  # [1/3] or np.linspace(1/3, 10/3, num=10)  # Target

    def __init__(self, J, gamma, epsilon, D, penalty, **kwargs):
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
        # self.rho = self.lambda_ / (self.s*self.mu)
        # target time list
        self.t = kwargs.get('t', np.random.choice(self.TARGET, self.J))
        self.weight = kwargs.get('weight', [1]*self.J)
        self.gamma = gamma  # Discritization parameter
        self.epsilon = epsilon  # Small number
        self.penalty = penalty  # Penalty never starting service
        self.D = D  # Cap on time
        self.P_xy = self.trans_prob()

    def init_lambda(self, rho):
        """Determine arrival rate based on desired load."""
        weight = np.random.uniform(self.imbalance_MIN,
                                   self.imbalance_MAX, self.J)
        return self.mu * (self.s*rho*weight/sum(weight))  # mu * (queue load)

    def trans_prob(self):
        """Docstring."""
        P_xy = np.zeros((self.J, self.D+1, self.D+1))  # class i in J, p_{x,y}
        gamma = self.gamma
        for i in range(self.J):  # For every class
            lambda_ = self.lambda_[i]
            for x in range(self.D+1):
                for y in range(1, x):
                    P_xy[(i, x, y)] = (gamma / (lambda_ + gamma))**(x-y) * \
                        lambda_ / (lambda_ + gamma)
                P_xy[(i, x, 0)] = (gamma / (lambda_ + gamma))**x
        return P_xy

    def reset(self):
        """Reset to empty state."""
        return np.zeros([self.J, 2])  # state [x,s]

    def get_s(self):
        """Docstring."""
        return self.s  # Include 0 states

    def get_dim(self):
        """Docstring."""
        return (self.D+1, self.s+1)*self.J  # Include 0 states

    def get_pi_0(self, _s, rho, **kwargs):
        """Calculate pi(0)."""
        Js = kwargs.get('i', range(self.J))  # Classes
        lambda_ = self.lambda_[Js]; gamma = self.gamma
        a = _s*rho
        pi_0 = _s * np.exp(a) / a**_s * \
            gamma_fun(_s) * reg_up_inc_gamma(_s, a)
        pi_0 += (gamma + rho * lambda_)/gamma * (1 / (1 - rho))
        return 1 / pi_0

    def get_tail_prob(self, _s, rho, pi_0, **kwargs):
        """P(W>t)."""
        Js = kwargs.get('i', range(self.J))
        lambda_ = self.lambda_[Js]; mu = self.mu[Js]; t = self.t[Js]
        gamma = self.gamma
        tail_prob = pi_0 / (1 - rho) * \
            (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
            (1 - (_s*mu - lambda_) / (_s*mu + gamma))**(t*gamma)
        return tail_prob

    def server_allocation_cost(self, _s):
        "Docstring."
        rho = self.lambda_ / (_s*self.mu)
        rho[rho == 1] = 1 - self.epsilon  # Avoid dividing by 0
        pi_0 = self.get_pi_0(_s, rho)
        return sum(self.weight*self.get_tail_prob(_s, rho, pi_0))

    def server_allocation(self):
        """
        Input: lambda, mu, t, weight.
        Output: s_star.
        """
        a = self.lambda_/self.mu; t = self.t; J = self.J; s = self.s
        weight = self.weight
        weighted_load = t/sum(t) * weight/sum(weight) * a/sum(a)
        importance = weighted_load/sum(weighted_load)
        x0 = a + importance * (s - sum(a))
        lin_cons = optimize.LinearConstraint(np.ones([J, J]), a, [s]* J)
        result = optimize.minimize(self.server_allocation_cost, x0,
                                   constraints=lin_cons)
        return result.x

    def c(self, x, i):
        """Input x must be a numpy array."""
        return ((self.penalty if x == self.D else 0) +
                (self.weight[i] if x > self.t[i]*self.gamma else 0))

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
