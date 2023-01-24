"""
Created on 19-3-2020.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)

Description: The class with the MDP environment.
"""

import numpy as np
from scipy.special import factorial as fac


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


class TimeConstraintEDs_int(Environment):
    """Environment of a Time Constraint Emergency Department."""

    def __init__(self, env):
        """Create all variables describing the environment."""
        self.J = env.J  # Classes
        self.s = env.s  # Servers
        self.mu = env.mu  # Service rate list
        self.lambda_ = env.lambda_  # Arrival rate list
        self.t = env.t  # target time list
        self.weight = env.weight  # Weight
        self.gamma = env.gamma  # Discritization parameter
        self.epsilon = env.epsilon  # Small number
        self.penalty = env.penalty  # Penalty never starting service
        self.D = env.D  # Cap on time
        self.P_xy = env.P_xy
        self.dim = env.dim
        self.tau = env.tau
        self.trace = env.trace
        self.print_modulo = env.print_modulo

    def init_lambda(self, rho):
        """Determine arrival rate based on desired load."""
        weight = np.random.uniform(self.imbalance_MIN,
                                   self.imbalance_MAX, self.J)
        return self.mu * (self.s*rho*weight/sum(weight))  # mu * (queue load)

    def reset(self):
        """Reset to empty state."""
        return np.zeros([self.J, 2])  # state [x,s]

    def get_dim(self):
        """Docstring."""
        return (self.D+1, self.s+1)*self.J  # Include 0 states

    def get_pi_0(self, _s, rho, **kwargs):
        """
        Calculate pi(0) for every class or just i.
        """
        Js = kwargs.get('i', range(self.J))  # Classes
        lambda_ = self.lambda_[Js]; gamma = self.gamma
        if 'i' in kwargs:
            _s = np.array([_s]); rho = np.array([rho]); Js = [0]; pi_0 = [0]
        else:
            pi_0 = np.zeros(len(Js))
        for i in Js:  # For every class
            k = np.array(range(int(_s[i])))
            pi_0[i] += sum((_s[i]*rho[i])**k / fac(k))
        pi_0 += (_s*rho)**_s / fac(_s) * \
            (gamma + rho * lambda_)/gamma * (1 / (1 - rho))
        return (1 / pi_0) * (_s * rho)**_s / fac(_s)

    def get_tail_prob(self, _s, rho, pi_0, **kwargs):
        Js = kwargs.get('i', range(self.J))  # Classes
        lambda_ = self.lambda_[Js]; mu = self.mu[Js]; t = self.t[Js]
        gamma = self.gamma
        tail_prob = pi_0 / (1 - rho) * \
            (lambda_ + gamma) / (gamma + lambda_ * pi_0) * \
            (1 - (_s*mu - lambda_) / (_s*mu + gamma))**(t*gamma)
        return tail_prob

    def server_allocation(self):
        t = self.t; weight = self.weight; s = self.s
        a = self.lambda_/self.mu
        lower_bound = np.maximum(a, 1)
        assert sum(lower_bound) <= s, "Lower bound cannot be satisfied with s integer."
        weighted_load = t/sum(t) * weight/sum(weight) * a/sum(a)
        importance = weighted_load/sum(weighted_load)
        sol = lower_bound + importance * (s - sum(lower_bound))
        sol_int = np.floor(sol)
        while(s - sum(sol_int) >= 1):  # While enough room to add a server.
            # Based on importance, +1 server for max
            sol_int[np.argmax(importance*1/sol_int)] += 1
        return sol_int

    def c(self, x, i):
        """Input x must be a numpy array."""
        return ((self.penalty if x == self.D else 0) +
                (self.weight[i] if x > self.t[i]*self.gamma else 0))

    def V_to_memory(self, _s, i, method, s_star):  # _s Extra
        """
        Calculate V for a single queue for all x = -s, ..., 0, ..., D.

        Only call this function once per class for effeciency.
        Handles s > s_star
        (if s > s_star, V(x)=0 for x = -s, ..., 0. As those states are
         never visited.)
        """
        lambda_ = self.lambda_[i]; mu = self.mu[i]; t = self.t[i]
        weight = self.weight[i]; gamma = self.gamma
        rho = lambda_ / (_s*mu)
        _s = int(_s)

        # Extra, calculate once, use pi_0[i], g[i], ...
        pi_0 = self.get_pi_0(_s, rho, i=i)
        tail_prob = self.get_tail_prob(_s, rho, pi_0, i=i)
        # Scale to get average reward
        g = tail_prob * (lambda_ + pi_0 * lambda_**2 / gamma)

        V = np.zeros(_s+1+self.D)  # V(-s) = 0, reference state
        if((method != 1) & (_s == s_star[i])):  # Extra
            # V(x), x<=0, Precalculate elements of double sum
            A = np.delete(np.indices((_s+1, _s+1)), 0, 1)  # Indices Matrix i,j
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp = fac(A[0] - 1) / fac(A[0] - A[1] - 1) * (mu/lambda_)**A[1]
            # Double sum
            for k in range(1, _s+1):
                V[k] = V[k-1] + sum(tmp[k-1, 0:k-1+1])
            V = g/lambda_*V  # Solve with self

        x = np.array(range(1, self.D+1))
        tmp_frac = (_s*mu + gamma) / (lambda_ + gamma)
        V[x+_s] = V[_s] + g / (gamma*_s*mu*(1 - rho)**2) * \
            (lambda_*x*(rho - 1) + (lambda_+gamma) * tmp_frac**x -
             (lambda_ + gamma))
        k = np.array(range(_s-1+1))
        V[x+_s] += g / (lambda_*gamma*(rho-1)) * \
            (lambda_ - gamma - gamma/rho *
             sum(fac(_s-1) / fac(_s-k-1) * (mu / lambda_)**k)) * \
            (-rho + (lambda_ + gamma) / (_s*mu + gamma) * tmp_frac**x)
        # -1_{x > t*gamma}[...]
        alpha = np.floor(t*gamma+1).astype(int)
        x = np.array(range(alpha, self.D+1))
        V[x+_s] -= weight/(gamma * (1 - rho)**2) * \
            ((lambda_*(x - t*gamma - 1)*(rho - 1) - (lambda_ + gamma)) +
             (lambda_ + gamma) * tmp_frac**(x-t*gamma-1))
        return V

    def policy_improvement(self, V, **kwargs):
        """
        Determine policy based on value.
        Calculate per state best action (policy) by one-step lookahead.
        """
        J = self.J; P_xy = self.P_xy
        policy = kwargs.get('policy', np.empty(self.dim, dtype=int))
        unstable = False

        it = np.nditer(policy, flags=['multi_index']); pointer = 0
        while not it.finished:
            multi_state = it.multi_index; it.iternext()
            if(pointer % self.print_modulo == 0):
                print("Policy Improvement:", multi_state); pointer += 1
            pointer += 1
            states = np.array(multi_state).reshape(J, 2)
            # If more servers used than possible
            if(sum(states[:,1]) > self.s):
                continue
            # If no server free or no one waiting, no decision needed.
            if((sum(states[:,1]) == self.s) | (sum(states[:,0]) == 0)):
                policy[multi_state] = J
            else:
                action_values = np.zeros(J+1)
                for i in range(J):
                    x_i = multi_state[i*2]
                    if(x_i > 0):  # If someone of class i waiting
                        next_state = list(multi_state)
                        next_state[i*2] = list(range(x_i, -1, -1))
                        next_state[i*2+1] += 1  # s
                        action_values[i] = self.c(x_i, i) + \
                            np.sum(P_xy[i, x_i, range(x_i+1)] * \
                                   V[tuple(next_state)])
                    else:  # Else no one of class i to take into service
                        action_values[i] = np.inf
                action_values[J] = V[multi_state]  # do nothing
                pi = action_values.min()
                if(pi != policy[multi_state]):
                    unstable = True
                policy[multi_state] = pi
        if 'policy' in kwargs:
            return policy, unstable
        else:
            return policy