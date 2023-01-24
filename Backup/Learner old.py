# -*- coding: utf-8 -*-
"""
Created on 19-3-2020

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)

Description: The class containing learner functies to take actions and learn
from experience. (The learner is also known as the agent or actor.)

The learner functions implemented are:
    - Value Iteration
    - Policy Iteration (PI)
    - One Step Policy Iteration
"""

import numpy as np
import timeit
from sys import exit

class Learner():
    """The learner's interface."""

    def __init__(self):
        self.name = 'name'
        pass

    def act(self, state):
        """Docstring."""
        pass

    def learn(self, experience):
        """Docstring."""
        pass


class OneStepPolicyImprovement(Learner):
    """Doing one step policy improvement."""

    def __init__(self, approx_method, env):
        self.name = 'One-step Policy Improvement'
        self.env = env; self.J = env.J; self.D = env.D; self.s = env.s
        self.approx_method = approx_method
        self.dim = self.env.dim
        self.policy = np.empty(self.dim, dtype=int)  # policy
        self.V_app = np.zeros(self.dim)
        self.V = np.zeros(self.dim)
        self.g = 0
        self.trace = env.trace; self.print_modulo = env.print_modulo

    def one_step_policy_improvement(self):
        """one step of policy improvement."""
        if self.trace: print('Starting One-step Policy improvement.')
        if self.trace: start = timeit.default_timer()
        s_star = self.env.server_allocation()
        self.approx_V(s_star)
        self.policy = self.env.policy_improvement(self.V_app)
        if self.trace:
            time = timeit.default_timer()-start
            print("Time: ", int(time/60), ":", int(time - 60*int(time/60)))

    def get_g(self):
        """
        Get average reward g.

        V is mutated within policy_evaluation."""
        self.g = self.env.policy_evaluation(self.policy, self.V)
        return self.g

    def approx_V(self, s_star):
        """Approximation of value function"""
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


class ValueIteration(Learner):
    """Doing Value Iteration."""

    def __init__(self, env, timeout):
        self.name = 'Value Iteration'
        self.env = env; self.J = env.J; self.D = env.D; self.s = env.s
        self.dim = self.env.dim; self.tau = self.env.tau
        self.policy = np.empty(self.dim, dtype=int)
        self.V = self.env.invalid_entries()  # V_{t-1}
        self.V_t = self.V.copy()  # V_{t}
        self.W = self.V.copy()
        self.g = 0
        self.P_xy = self.env.P_xy
        self.timeout = timeout

    def value_iteration(self):
        """Docstring."""
        env = self.env; counter = 0
        env.timer(True, self.name)
        while True:
            # One-step lookahead.
            W = env.W(self.V, self.W)
            print(counter)
            env.one_step_lookahead(self.V_t, self.V, self.W, self)
            print("W", self.W)  # TODO
            # Stopping condition
            if env.convergence(self.V_t, self.V, counter):
                break
            # Save and Rescale the value function
            self.V = self.V_t.copy()  # - self.V_t[(0,0)*self.J]
            counter += 1
        env.timer(False, self.name)

    # def calculate_policy(self):
    #     """Docstring."""
    #     self.policy = self.env.policy_improvement(self.V_app)


class PolicyIteration(Learner):
    """policy iteration."""

    def __init__(self, env, timeout):
        self.name = 'Policy Iteration'
        self.env = env; self.J = env.J; self.D = env.D; self.s = env.s
        self.dim = self.env.dim; self.tau = self.env.tau
        self.policy = np.full(self.dim, self.J, dtype=int)
        self.V = self.env.invalid_entries()  # V_{t-1}
        self.V_t = self.V.copy()  # V_{t}
        self.W = self.V.copy()
        self.g = 0
        self.P_xy = self.env.P_xy
        self.timeout = timeout


    def value_iteration(self):
        """Docstring."""
        env = self.env; counter = 0
        env.timer(True, self.name)
        while True: # Update each state.
            env.calculate_W(self.V, self.W)
            # print("counter", self.counter)
            print(counter)
            env.one_step_lookahead(self.V_t, self.V, self.W, self)
            print("W", self.W)  # TODO
            # Stopping condition
            if env.convergence(self.V_t, self.V, counter):
                break
            # Save and Rescale the value function
            self.V = self.V_t.copy()  # - self.V_t[(0,0)*self.J]
            counter += 1
        env.timer(False, self.name)


    def policy_iteration(self):
        """Policy Iteration."""
        env = self.env; counter = 0; unstable = True
        env.timer(True, self.name)
        while unstable:
            # Policy Evaluation
            inner_counter = 0
            while True:
                env.calculate_W(self.V, self.W)
                env.policy_evaluation(self.V_t, self.V, self.W, self)
                # Stopping condition
                if env.convergence(self.V_t, self.V, inner_counter):
                    # if self.trace: print("delta: ", delta,
                    #      "Inner iterations: ", inner_counter)
                    break
                # Save and Rescale the value function
                self.V = self.V_t.copy()  # - self.V_t[(0,0)*self.J]
                inner_counter += 1
            # Policy Improvement
            self.policy_t, unstable = \
                self.env.policy_improvement(self.V, policy=self.policy)
            counter += 1
            # if self.trace: print("Iterations: ", counter)
        env.timer(False, self.name)

    def policy_evaluation(self, multi_state):
        """Policy evaluation of given state."""
        lambda_ = self.env.lambda_; mu = self.env.mu; gamma = self.env.gamma
        # Determine value by one-step lookahead
        V_t = 0
        dummy = self.tau
        for i in range(self.J):
            x_i = multi_state[i*2]; s_i = multi_state[i*2+1]
            next_state = list(multi_state)
            next_state[i*2] = min(x_i + 1, self.D)
            g = gamma if(x_i > 0) else lambda_[i]
            dummy -= g
            V_t += g * self.W(multi_state, i)
            if(s_i > 0):
                next_state = list(multi_state)
                next_state[i*2+1] -= 1
                V_t += s_i * mu[i] * self.W(tuple(next_state), i)
                dummy -= s_i*mu[i]
        V_t += dummy * self.V[multi_state]
        return V_t

    def W(self, multi_state, i):
        """
        Return value given policy.
        """
        states = np.array(multi_state).reshape(self.J, 2)
        if((self.policy[multi_state] == self.J) |
           (sum(states[:,1]) == self.s) | (sum(states[:,0]) == 0)):
            self.policy[multi_state] == self.J
            return self.V[multi_state] + max(self.env.c_D(states[:,0]))
        w = 0
        x_i = multi_state[i*2]
        if(x_i > 0):  # If someone of class i waiting
            next_state = list(multi_state)
            next_state[i*2] = list(range(x_i, -1, -1))
            next_state[i*2+1] += 1  # s
            w = self.env.c(x_i, i) + \
                np.sum(self.P_xy[i, x_i, range(x_i+1)] * \
                       self.V[tuple(next_state)])
        else:  # Else no one of class i to take into service
            # action: do nothing
            w = self.V[multi_state] + self.c_D(states[:,0])
        return w
