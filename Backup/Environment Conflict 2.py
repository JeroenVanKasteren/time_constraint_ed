"""
Created on 19-3-2020.

@author: Jeroen van Kasteren (j.van.kasteren@vu.nl)

Description: The class with the MDP environment.

mu, lambda and similar vectors are assumed to be numpy arrays.
"""

import numpy as np
import itertools
from scipy import optimize
from scipy.integrate import quad_vec
from scipy.special import gamma as gamma_fun
from scipy.special import gammaincc as reg_up_inc_gamma
from sys import exit
import timeit


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
    INVALID = -np.inf

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
        print("a:", np.round(self.mu/self.lambda_, 4))
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
        t_temp = np.copy(t); t_temp[t==0] = 1/10  # Avoid dividing by zero.
        weighted_load = t_temp/sum(t_temp) * weight/sum(weight) * a/sum(a)
        importance = weighted_load/sum(weighted_load)
        x0 = a + importance * (s - sum(a))
        lin_cons = optimize.LinearConstraint(np.ones([J, J]), a, [s]* J)
        result = optimize.minimize(self.server_allocation_cost, x0,
                                   constraints=lin_cons)
        return result.x

    def c(self, x, i):
        """Input x must be a numpy array."""
        return np.where(x > self.alpha[i], self.weight[i], 0)

    def c_D(self, x):
        """Docstring."""
        return self.penalty if any(np.array(x) == self.D) else 0

    def c_matrix(self):
        """Docstring."""
        penalty_m = np.zeros([self.D+1]*self.J)
        for i in range(self.J):
            state = [slice(None)]*(self.J-1)
            state.insert(i, self.D)
            penalty_m[tuple(state)] = self.penalty
        return penalty_m

    def invalid_entries(self, *args):
        """Marks all invalid entries of table."""
        it = np.nditer(args[0], flags=['multi_index'])
        while not it.finished:
            multi_state = it.multi_index; it.iternext()
            for arg in args:
                if(sum(multi_state[slice(1, self.J*2, 2)]) > self.s):
                    arg[multi_state] = self.INVALID

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

    def setup_iterate(self, matrix):
        """Docstring."""
        self.counter += 1
        self.delta_max = 0; self.delta_min = 0; self.pointer = 0
        return np.nditer(matrix, flags=['multi_index'])

    def iterate(self, iterator):
        """Docstring."""
        while True:
            multi_state = iterator.multi_index; iterator.iternext()
            if(self.V_t[multi_state] == self.INVALID):
                continue
            if(self.pointer % self.print_modulo == 0):
                if self.trace: print("VI:", multi_state)
            self.pointer += 1
            return multi_state

    def convergence(self, v_t, v):
        """Docstring."""
        diff = np.abs(v_t - v)
        self.delta_max = max(self.delta_max, diff)
        self.delta_min = min(self.delta_min, diff)

    def converged(self):
        """Docstring."""
        _converged = self.delta_max - self.delta_min < self.epsilon
        if(_converged):
            if self.trace: print("delta: ", self.delta_max - self.delta_min,
                                 "g: ", self.delta_max, self.delta_min,
                                 ", Iterations: ", self.counter)
            elif self.counter > self.timeout:
                if self.trace:
                    print('Value Iteration reached timeout')
                    print("delta:",np.round(self.delta_max-self.delta_min,2),
                          ", Iterations: ", self.counter)
        return _converged

    def combinations(self):
        """Calculate all valid server and waiting time combinations."""
        s_states = itertools.product(range(self.s+1), repeat=self.J)
        s_states = np.array(list(s_states))
        s_states = s_states[np.sum(s_states, axis=1) <= self.s]
        self.s_states = s_states
        self.D_states = np.array(list(itertools.product(range(self.D+1),
                                                        repeat=self.J)))
        self.D_1_states = np.array(list(itertools.product(range(self.D+1),
                                                          repeat=self.J-1)))

    def calculate_W(self, W, V):
        """Docstring."""
        J = self.J; D = self.D
        for s_state in self.s_states:
            states = np.array([[slice(None)]*J, s_state]).T.reshape(-1)
            penalty_m = self.c_matrix()
            if sum(s_state) == self.s:
                W[states] = V[states] + penalty_m
                continue
            w = np.zeros([J]+[D+1]*J)  # w(i, x)
            for i in range(J):
                for x in self.D_1_states:
                    x = list(x)
                    x.insert(i, slice(None))
                    next_state = np.array([x, s_state]).T.reshape(-1)
                    next_state[i*2] = slice(D, None, -1)  # x_i - (x_i - y)
                    next_state[i*2+1] += 1  # s_i + 1
                    w[tuple([i]+x)] = self.c(np.arange(D+1), i) +\
                        self.P_xy[i, slice(None), slice(None)] *\
                        V[next_state]
                    W[states] = np.min(
                        np.concatenate((w, [V[states] + penalty_m]), axis=0),
                        axis=0)


    def update(self, multi_state, value_state):
        """One-step lookahead."""
        value = 0; dummy = self.tau
        for i in range(self.J):
            x_i = multi_state[i*2]; s_i = multi_state[i*2+1]
            next_state = list(multi_state)
            next_state[i*2] = min(x_i + 1, self.D)
            f = self.gamma if(x_i > 0) else self.lambda_[i]
            dummy -= f
            value += f * self.W(multi_state)
            if(s_i > 0):
                next_state = list(multi_state)
                next_state[i*2+1] -= 1
                value += s_i * self.mu[i] * self.W(tuple(next_state))
                dummy -= s_i * self.mu[i]
        value += dummy * value_state
        return value / self.tau

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

    def policy_improvement(self, V, **kwargs):
        """
        Determine policy based on value.
        Calculate per state best action (policy) by one-step lookahead.
        """
        J = self.J; P_xy = self.P_xy
        policy = kwargs.get('policy', np.zeros(self.dim, dtype=int))
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
                # do nothing
                action_values[J] = V[multi_state] + max(self.c_D(states[:,0]))
                pi = action_values.argmin()
                if(pi != policy[multi_state]):
                    unstable = True
                policy[multi_state] = pi
        if 'policy' in kwargs:
            return policy, unstable
        else:
            return policy

    def print_g(self, V):
        multi_state = list((0,0)*self.J)
        print("g:", self.get_g(tuple(multi_state), V))
        for _ in range(5):
            servers_left = self.s + 1
            for i in range(self.J):
                multi_state[i*2] = np.random.randint(self.D+1)
                multi_state[i*2+1] = np.random.randint(servers_left)
                servers_left -= multi_state[i*2+1]
            print("g:", self.get_g(tuple(multi_state), V))

    def get_g(self, multi_state, V):
        """Docstring."""
        lambda_ = self.lambda_; mu = self.mu; gamma = self.gamma
        # Determine value by one-step lookahead
        V_t = 0
        dummy = 0
        for i in range(self.J):
            x_i = multi_state[i*2]; s_i = multi_state[i*2+1]
            next_state = list(multi_state)
            next_state[i*2] = min(x_i + 1, self.D)
            f = gamma if(x_i > 0) else lambda_[i]
            dummy -= f
            V_t += f * self.W(multi_state, V)
            if(s_i > 0):
                next_state = list(multi_state)
                next_state[i*2+1] -= 1
                V_t += s_i * mu[i] * self.W(tuple(next_state), V)
                dummy -= s_i*mu[i]
        V_t += dummy * V[multi_state]
        return V_t

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

    def policy_evaluation(self, policy, V):
        counter = 0
        V_t = np.zeros(self.dim)
        while True:
            delta = 0; counter += 1
            # Update each state.
            it = np.nditer(V, flags=['multi_index']); pointer = 0
            while not it.finished:
                multi_state = it.multi_index; it.iternext()
                if(pointer % self.print_modulo == 0):
                    if self.trace: print("Policy evaluation", multi_state)
                pointer += 1
                states = np.array(multi_state).reshape(self.J, 2)
                # If more servers used than possible
                if(sum(states[:,1]) > self.s):
                    continue
                V_t[multi_state] = self.evaluate_state(multi_state, policy, V)
                delta = max(delta, np.abs(V_t[multi_state] - V[multi_state]))
                if(delta > 1e4):
                    print("Delta:", np.round(delta,2),  # TODO
                      "state", np.array(multi_state),
                      "policy", policy[multi_state],
                      "V_t", np.round(V_t[multi_state],2),
                      "V old", np.round(V[multi_state],2))
                if (delta > 1e8):
                    print(multi_state)  # TODO
                    print("V:", np.round(V[multi_state],2))
                    print("V_t:", np.round(V[multi_state],2))
                    print(V)
                    exit(0)
                # Update the value function
                V = V_t - V_t[(0,0)*self.J]
            if self.trace: print("delta: ", delta,
                 "Inner iterations: ", counter)
            # Stopping condition
            if delta < self.epsilon:
                break
            if(counter > self.timeout):
                if self.trace: print('Policy Evaluation reached timeout')
                break

    def evaluate_state(self, multi_state, policy, V):
        """Policy evaluation of given state."""
        lambda_ = self.lambda_; mu = self.mu; gamma = self.gamma
        # Determine value by one-step lookahead
        V_t = 0
        dummy = self.tau
        for i in range(self.J):
            x_i = multi_state[i*2]; s_i = multi_state[i*2+1]
            next_state = list(multi_state)
            next_state[i*2] = min(x_i + 1, self.D)
            g = gamma if(x_i > 0) else lambda_[i]
            dummy -= g
            V_t += g * self.W_i(multi_state, i, policy, V)
            if(s_i > 0):
                next_state = list(multi_state)
                next_state[i*2+1] -= 1
                V_t += s_i * mu[i] * self.W_i(tuple(next_state), i, policy, V)
                dummy -= s_i*mu[i]
        V_t += dummy * V[multi_state]
        return V_t


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