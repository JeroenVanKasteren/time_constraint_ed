"""
Learner functies to take actions and learn from experience.

(The learner is also known as the agent or actor.)
The learner functions implemented are:
    - Value Iteration
    - Policy Iteration (PI)
    - One Step Policy Iteration

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)
"""

import numpy as np
from numpy import array, arange, zeros
from numba import njit
from scipy import optimize
from scipy.integrate import quad_vec
from scipy.special import gamma as gamma_fun, gammaincc as reg_up_inc_gamma

class OneStepPolicyImprovement():
    """Doing one step policy improvement."""

    def __init__(self, env, approx_method):
        self.name = 'One-step Policy Improvement'
        self.V_app = np.zeros(env.dim)
        self.V = zeros(env.dim)  # V_{t-1}
        self.W = zeros(env.dim)
        self.Pi = -np.ones(env.dim)
        self.g = 0
        self.approx_method = approx_method

    def one_step_policy_improvement(self, env):
        """One Step of Policy Improvement."""
        env.timer(True, 'One-step Policy improvement', env.trace)
        s_star = env.server_allocation()
        self.approx_V(s_star)
        self.V, self.Pi, _ = env.policy_improvement(self.V_app, self.Pi)
        env.timer(False, 'One-step Policy improvement', env.trace)

    def get_g(self):
        """
        Get average reward g.

        V is mutated within policy_evaluation.
        """
        self.g = self.env.policy_evaluation(self.policy, self.V)
        return self.g

    def approx_V(self, s_star):
        """Approximation of value function."""
        J = self.J; s = self.s; D = self.D; method = self.approx_method
        s_n = np.floor(s_star).astype(int)
        """
        Create a list V_memory with V_ij(x), i=class, j=#servers for all x.
        Note only j = s*_i, ..., s will be filled, rest zero
        """
        if(method == 3):
            V_memory = np.zeros([J, s+1, s+1+D])  # +1 for state 0
        else:
            V_memory = np.zeros([J, 1, s+1+D])  # +1 for state 0

        for i in range(J):
            if self.trace: print("Value approximation to memory class, ", i)
            V_memory[i, 0, slice(s-s_n[i], s+1+D)] = \
                self.env.V_to_memory(s_star[i], i, method, s_star)
            if(method == 3):
                for s_i in range(s_n[i]+1, s+1):  # s_i > s*_i
                    V_memory[i, s_i, slice(s-s_i, s+1+D)] = \
                        self.env.V_to_memory(s_i, i, method, s_star)
        # Loop over every multi-class state
        it = np.nditer(self.V_app, flags=['multi_index']); pointer = 0
        while not it.finished:
            multi_state = it.multi_index; it.iternext()
            if(pointer % self.print_modulo == 0):
                if self.trace: print("V(x,s) = Vapp:", multi_state)
            pointer += 1
            states = np.array(multi_state).reshape(J, 2)
            # If more servers used than possible
            if(sum(states[:,1]) > self.s):
                continue
            V_multi = 0  # Value in multi-class state
            # Sum every single-class state to approx multi-class state
            for i, state in enumerate(states):
                x_i = state[0]; s_i = state[1]
                if(method == 1):
                    V_multi += V_memory[i, 0, x_i+s]
                elif(method == 2):
                    if((s_i > s_star[i]) | (x_i > 0)):
                        V_multi += V_memory[i, 0, x_i+s]
                    else:
                        V_multi += V_memory[i, 0, s_i-s_n[i]+x_i+s]
                else:  # Method 3
                    if(s_i > s_star[i]):
                        V_multi += V_memory[i, s_i, x_i+s]
                    elif(x_i > 0):
                        V_multi += V_memory[i, 0, x_i+s]
                    else:
                        V_multi += V_memory[i, 0, s_i-s_n[i]+x_i+s]
            self.V_app[multi_state] = V_multi

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
        """Docstring."""
        rho = self.lambda_ / (_s*self.mu)
        rho[rho == 1] = 1 - self.epsilon  # Avoid dividing by 0
        pi_0 = self.get_pi_0(_s, rho)
        return sum(self.weight*self.get_tail_prob(_s, rho, pi_0))

    def server_allocation(self, env):
        """Docstring."""
        J = env.J
        a = env.lambda_/env.mu
        t = env.t

        _t = 1/100 if t == 0 else t  # Avoid dividing by zero.
        weighted_load = _t/sum(_t) * env.c/sum(env.c) * a/sum(a)
        importance = weighted_load / sum(weighted_load)
        x0 = a + importance * (env.s - sum(a))
        lin_cons = optimize.LinearConstraint(np.ones([J, J]), a, [env.s]* J)
        result = optimize.minimize(self.server_allocation_cost, x0,
                                   constraints=lin_cons)
        return result.x

class ValueIteration():
    """Value Iteration."""

    def __init__(self, env):
        self.name = 'Value Iteration'
        self.V = zeros(env.dim)  # V_{t-1}
        self.W = zeros(env.dim)
        self.Pi = -np.ones(env.dim)
        self.g = 0

    def value_iteration(self, env):
        """Docstring."""
        counter = 0
        env.timer(True, self.name, env.trace)
        while True:  # Update each state.
            V, W = self.W(self.V, self.W)
            V_t = env.V(self.V, self.W)

            if(env.convergence(V_t, self.V, counter)):
                break  # Stopping condition

            # Rescale and Save V_t
            self.V = V_t - V_t[tuple([0]*(self.J*2))]
            counter += 1
        env.timer(False, self.name, env.trace)

    def policy(self, env):
        """Determine policy via Policy Improvement."""
        self.V, self.Pi, _ = env.policy_improvement(self.V, self.Pi)

    @njit
    def W(env, V, W):
        """W."""
        J = env.J
        sizes = env.sizes
        V = V.reshape(env.size)
        W = W.reshape(env.size)
        for s in env.s_states:
            for x in env.x_states:
                state = np.sum(x*sizes[0:J] + s*sizes[J:J*2])
                W[state] = V[state] + (env.P if np.any(x == env.D) else 0)
                if np.sum(s) < env.S:
                    for i in arange(J):
                        if(x[i] > 0):  # If someone of class i waiting
                            value = env.c[i] if x[i] > env.alpha[i] else 0
                            for y in arange(x[i] + 1):
                                next_x = x.copy()
                                next_x[i] = y
                                next_s = s.copy()
                                next_s[i] += 1
                                next_state = np.sum(next_x*sizes[0:J] + \
                                                    next_s*sizes[J:J*2])
                                value += env.P_xy[i, x[i], y] * V[next_state]
                            W[state] = array([value, W[state]]).min()
        return V.reshape(env.dim), W.reshape(env.dim)

class PolicyItertion():
    """Policy Iteration."""

    def __init__(self, env):
        self.name = 'Policy Iteration'
        self.env = env
        self.V = np.zeros(env.dim)  # V_{t-1}
        self.W = np.zeros(env.dim)
        self.Pi = -np.ones(env.dim)
        self.g = 0

    def policy_iteration(self, env):
        """Docstring."""
        counter = 0
        unstable = True

        env.timer(True, self.name, env.trace)
        while unstable:
            # Policy Evaluation
            inner_counter = 0
            while True:
                self.V, self.W, self.Pi = env.W(self.V, self.W, self.Pi)
                V_t = env.V(self.V, self.W)

                if (env.convergence(V_t, self.V, inner_counter)):
                    break  # Stopping condition

                # Rescale and Save V_t
                self.V = V_t - V_t[tuple([0]*(self.J*2))]
                inner_counter += 1

            # Policy Improvement
            self.V, self.Pi, unstable = env.policy_improvement(self.V,
                                                               self.Pi)
            if counter > env.max_iter:
                break
            counter += 1
        self.env.timer(False, self.name, env.trace)
