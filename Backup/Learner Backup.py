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

    def __init__(self, J, D, approx_method, env):
        self.name = 'One-step Policy Improvement'
        self.J = J; self.D = D; self.env = env
        self.s = env.get_s()
        self.approx_method = approx_method
        self.dim = self.env.get_dim()
        self.policy = np.zeros(self.dim, dtype=int)  # policy

    def one_step_policy_improvement(self):
        """one step of policy improvement."""
        J = self.J
        s_star = self.env.server_allocation()
        P_xy = self.env.P_xy
        V_app = self.approx_V(s_star)
        # Calculate per state best action (policy) by one-step lookahead
        it = np.nditer(self.policy, flags=['multi_index']); pointer = -1
        while not it.finished:
            multi_state = it.multi_index; it.iternext()
            if(multi_state[0] > pointer):
                print(multi_state); pointer += 1
            states = np.array(multi_state).reshape(J, 2)
            # If no server free or no one waiting, no decision needed.
            if((sum(states[:,1]) >= self.s) | (sum(states[:,0]) == 0)):
                self.policy[multi_state] = J
            else:
                action_values = np.zeros(J+1)
                for i in range(J):
                    x_i = multi_state[i*2]
                    if(x_i > 0):  # If someone of class i waiting
                        next_state = list(multi_state)
                        next_state[i*2] = list(range(x_i, -1, -1))
                        next_state[i*2+1] += 1  # s
                        # Missing code that sum of s < s
                        action_values[i] = self.env.c(x_i, i) + \
                            np.sum(P_xy[i, x_i, range(x_i+1)] * \
                                   V_app[tuple(next_state)])
                    else:  # Else no one of class i to take into service
                        action_values[i] = np.inf
                action_values[J] = V_app[multi_state]  # do nothing
                self.policy[multi_state] = np.argmin(action_values)

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
            print("Value approximation to memory class, ", i)
            V_range = range(s-s_n[i], s+1+D)
            V_memory[i, 0, V_range] = \
                self.env.V_to_memory(s_star[i], i, method, s_star)
            if(method == 3):
                for s_i in range(s_n[i]+1, s+1):  # s_i > s*_i
                    V_range = range(s-s_i, s+1+D)
                    V_memory[i, s_i, V_range] = \
                        self.env.V_to_memory(s_i, i, method, s_star)
        # Loop over every multi-class state
        V_app = np.zeros(self.dim)  # Value function
        it = np.nditer(V_app, flags=['multi_index']); pointer = -1
        while not it.finished:
            multi_state = it.multi_index
            if(multi_state[0] > pointer):
                print(multi_state); pointer += 1
            V_multi = 0  # Value in multi-class state
            states = np.array(multi_state).reshape(J, 2)
            # Sum every single-class state to approx multi-class state
            for i, state in enumerate(states):
                x_i = state[0]; s_i = state[1]
                if(method == 1):
                    V_multi += V_memory[i, 0, x_i+s]
                elif(method == 2):
                    if(s_i > s_star[i] | x_i > 0):
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
            V_app[multi_state] = V_multi
            it.iternext()
        return V_app


# class PolicyIteration(Learner):
#     """policy iteration."""

#     def __init__(self, env):
#         self.name = 'Policy Iteration'
#         self.env = env
#         dim = self.env.get_dim()
#         self.V = np.zeros(dim)  # Value function
#         self.policy = np.zeros(dim)  # policy

#     def policy_iteration(self, optimistic_start, SERVER_INDEP):
#         """Policy Iteration."""
#         pass

#     def W_VI(self, x, s, V_t):
#         """Return value."""
#         if(sum(s) == self.s):
#             return V_t[(:,x,s)]
#         cost = c(x)
#         w = np.zeros(self.N+1)
#         for i in range(self.N):
#             _x = x[i]
#             y = range(_x)
#             w[i] = cost[i] + np.dot(P_xy[(i, _x, y)], V_t[(i, y, s[i]+1)]
#         w[self.N] = V_t[state]
#         _min = w.min()
#         indeces = np.where(w == _min)
#         return (_min, np.random.choice(index))

#     def W_PI(self, V):
#         """Return policy."""
#         V = 1
#         w = np.zeros(self.N)
#         # for i in range(self.J):
#         return w


#         Vapp(x_i, s_i, s, lambda_, mu, t, D, gamma)
#         for()

#     def act(self, state):
#         pass

#     def learn(self, experience):
#         pass

# def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
#     # Initialize thel value function
#     V = np.zeros(env.nS)
#     # While our value function is worse than the threshold theta
#     while True:
#         # Keep track of the update done in value function
#         delta = 0
#         # For each state, look ahead one step at each possible action and next state
#         for s in range(env.nS):
#             v = 0
#             # The possible next actions, policy[s]:[a,action_prob]
#             for a, action_prob in enumerate(policy[s]):
#                 # For each action, look at the possible next states,
#                 for prob, next_state, reward, done in env.P[s][a]: # state transition P[s][a] == [(prob, nextstate, reward, done), ...]
#                     # Calculate the expected value function
#                     v += action_prob * prob * (reward + discount_factor * V[next_state]) # P[s, a, s']*(R(s,a,s')+γV[s'])
#                     # How much our value function changed across any states .
#             delta = max(delta, np.abs(v - V[s]))
#             V[s] = v
#         # Stop evaluating once our value function update is below a threshold
#         if delta < theta:
#             break
#     return np.array(V)


# def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
#     # Initiallize a policy arbitarily
#     policy = np.ones([env.nS, env.nA]) / env.nA

#     while True:
#         # Compute the Value Function for the current policy
#         V = policy_eval_fn(policy, env, discount_factor)

#         # Will be set to false if we update the policy
#         policy_stable = True

#         # Improve the policy at each state
#         for s in range(env.nS):
#             # The best action we would take under the currect policy
#             chosen_a = np.argmax(policy[s])
#             # Find the best action by one-step lookahead
#             action_values = np.zeros(env.nA)
#             for a in range(env.nA):
#                 for prob, next_state, reward, done in env.P[s][a]:
#                     action_values[a] += prob * (reward + discount_factor * V[next_state])
#             best_a = np.argmax(action_values)

#             # Greedily (max in the above line) update the policy
#             if chosen_a != best_a:
#                 policy_stable = False
#             policy[s] = np.eye(env.nA)[best_a]

#         # Until we've found an optimal policy. Return it
#         if policy_stable:
#             return policy, V

# def value_iteration(env, theta=0.0001, discount_factor=1.0):
#     # Look ahead one step at each possible action and next state (full backup)
#     def one_step_lookahead(state, V):
#         """
#         Helper function to calculate the value for all action in a given state.

#         Args:
#             state: The state to consider (int)
#             V: The value to use as an estimator, Vector of length env.nS

#         Returns:
#             A vector of length env.nA containing the expected value of each action.
#         """
#         A = np.zeros(env.nA)
#         for a in range(env.nA):
#             for prob, next_state, reward, done in env.P[state][a]:
#                 A[a] += prob * (reward + discount_factor * V[next_state])
#         return A

#     V = np.zeros(env.nS)
#     while True:
#         # Stopping condition
#         delta = 0
#         # Update each state...
#         for s in range(env.nS):
#             # Do a one-step lookahead to find the best action
#             A = one_step_lookahead(s, V)
#             best_action_value = np.max(A)
#             # Calculate delta across all states seen so far
#             delta = max(delta, np.abs(best_action_value - V[s]))
#             # Update the value function
#             V[s] = best_action_value
#         # Check if we can stop
#         if delta < theta:
#             break

#     # Create a deterministic policy using the optimal value function
#     policy = np.zeros([env.nS, env.nA])
#     for s in range(env.nS):
#         # One step lookahead to find the best action for this state
#         A = one_step_lookahead(s, V)
#         best_action = np.argmax(A)
#         # Always take the best action
#         policy[s, best_action] = 1.0

#     return policy, V

