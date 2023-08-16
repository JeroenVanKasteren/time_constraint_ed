"""
This file contains the simulation of the multi-class queueing system.
"""

import heapq as hq
import numba as nb
import numpy as np
from utils import TimeConstraintEDs as Env
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

# -------------------------- Value Iteration --------------------------------
pi_learner = PolicyIteration()
env = Env(J=2, S=4, gamma=10, D=50, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=10, print_modulo=100)

'''
he following functions from the heapq module are supported:

heapq.heapify()
heapq.heappop()
heapq.heappush()
heapq.heappushpop()
heapq.heapreplace()
heapq.nlargest() : first two arguments only
heapq.nsmallest() : first two arguments only
Note: the heap must be seeded with at least one value to allow its type to be inferred; heap items are assumed to be homogeneous in type.

import numba as nb
import heapq as hq

entry_type = nb.typeof((0.0, 0, 'event'))

@njit
def heapsort(iterable):
    time = 0
    heap = nb.typed.List.empty_list(entry_type)
    for i in range(len(iterable)):
        hq.heappush(heap, (iterable[i], 0, 'arrival'))
        time += iterable[i]
    return heap, time

x = nb.typed.List([1.232, 3.21, 5.21, 7.54, 9.765, 2.35, 4.85, 6.00, 8.1, 0.23])
print(heapsort(x))
'''
# Choice: N = # events to simulate or T = time to simulate.

@nb.njit
def simulate_multi_class_system(J, S, lab_i, mu_i, r_i, c_i, t_i, N):
    # generate arrival & service times
    rng = np.random.default_rng()  # add seed?
    arrival_times = np.empty((env.J, env.D), dtype=np.float32)
    service_times = np.empty((env.J, env.D), dtype=np.float32)
    for i in range(env.J):  # Does this vec. work in numba? Maybe put outside of function?
        arrival_times[i, :] = env.rng.exponential(1 / env.lab[i], N)
        service_times[i, :] = env.rng.exponential(1 / env.mu[i], N)

    # initialize the system state
    state = np.zeros(J+1, dtype=np.int32)
    total_reward = np.zeros(J+1)
    total_cost = np.zeros(J+1)

    # initialize the event list
    events = []

    # schedule the first arrival events
    for i in range(J):
        events.append((np.random.exponential(1/lambda_i[i]), i, 'arrival'))

    # run the simulation
    for t in range(T):
        # check if any service initiation events can occur
        if np.sum(state) < S:
            prob = r_i / np.sum(r_i)
            i = np.random.choice(J, p=prob)
            if state[i] > 0:
                state[i] -= 1
                total_reward[i] += r_i[i]
                events.append((t, i, 'departure'))

        # process the next event
        next_event_idx = np.argmin([event[0] for event in events])
        next_event = events.pop(next_event_idx)
        t_event = next_event[0]
        i_event = next_event[1]
        type_event = next_event[2]

        if type_event == 'arrival':
            # add the customer to the shortest queue for class i
            q = np.where(state[:J] == np.min(state[:J]))
            i_queue = np.random.choice(q[0])
            if state[i_queue] < S:
                state[i_queue] += 1
                events.append((t_event + np.random.exponential(1/lambda_i[i_event]), i_event, 'arrival'))
            else:
                events.append((t_event + np.random.exponential(1/lambda_i[i_event]), i_event, 'dropped'))

        elif type_event == 'departure':
            # remove the customer from the queue
            if state[i_event] > 0:
                state[i_event] -= 1
                if np.sum(total_reward[:J]) >= t_i[i_event]:
                    total_cost[i_event] += c_i[i_event]
                    t_i[i_event] *= 2

'''
Key things to consider:
The Numba type of a typed.List is a types.ListType.
One cannot call typeof from inside a jitted region, 
(Workaround, mytype = typeof(mytype_instance) globally and refer to mytype)
typed.List.empty_list and typed.Dict.empty take Numba types as arguments.
'''

import numba as nb
import heapq as hq

entry_type = nb.typeof((0.0, 0, 'event'))

@nb.njit
def heapsort(iterable):
    time = 0
    heap = nb.typed.List.empty_list(entry_type)
    for i in range(len(iterable)):
        hq.heappush(heap, (iterable[i], 0, 'arrival'))
        time += iterable[i]
    return heap, time

x = nb.typed.List([1.232, 3.21, 5.21, 7.54, 9.765, 2.35, 4.85, 6.00, 8.1, 0.23])
print(heapsort(x))
