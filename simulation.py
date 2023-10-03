"""
This file contains the simulation of the multi-class queueing system.
"""

# import argparse
import heapq as hq
import numba as nb
import numpy as np
import pandas as pd
# from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import OneStepPolicyImprovement as Ospi

# global constants
N = 10  # arrivals to simulate
batch_size = 100  # batch size for KPI
strategy = 'fcfs'  # 'fcfs' 'sdf' 'sdf_prior' 'cmu' 'ospi'

J = 3
S = 5
gamma = 30
D = 100
t = np.array([30, 60, 120])
c = np.array([1]*J)
r = np.array([1]*J)
# lab = np.array([14/60*0.5, 14/60*0.4, 14/60*0.1])
imbalance = np.array([0.5, 0.4, 0.1])
mu = np.array([1/10, 1/20, 1/30])
load = 0.75
env = Env(J=J, S=S, D=D, gamma=gamma, t=t, c=c, r=r, mu=mu, load=load,
          imbalance=imbalance)
# lab=lab, e=0.1, max_time=args.time,convergence_check=10)
lab = env.lab

p_xy = env.p_xy
regret = np.max(r) - r + c
cmu = c * mu


def generate_times(J, N, lab, mu):
    """Generate exponential arrival and service times."""
    arrival_times = nb.typed.List[np.float32]()  # +1, last arrival
    service_times = nb.typed.List[np.float32]()
    for i in range(J):
        arrival_times.append(nb.typed.List(env.rng.exponential(1 / lab[i], N + 1)))
        service_times.append(nb.typed.List(env.rng.exponential(1 / mu[i], N + 1)))
    return arrival_times, service_times


def get_v_app(env):
    """Get the approximate value function for a given state."""
    v = np.zeros((env.J, env.D + 1))
    for i in range(env.J):
        v[i, ] = Ospi.get_v_app_i(env, i)
    return v


heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = get_v_app(env)
arrival_times, service_times = generate_times(J, N, lab, mu)
kpi = np.zeros((N+1, 3))  # time, class, waited


# @nb.njit
def ospi(x, i):
    """One-step policy improvement.
    i indicate which class just arrived, i = J if no class arrived.
    """
    x = np.minimum(np.round(x / gamma), D)
    pi = J
    x_next = x if i == J else x + eye[i]
    v_sum = np.zeros(J)
    w_max = 0
    for j in range(J):
        w_max += v[j, x_next[j]]
        v_sum += v[j, x_next[j]]
        v_sum[j] -= v[j, x_next[j]]
    for j in range(J):  # Class to admit
        if (x[j] > 0) or (j == i):
            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
            w += p_xy[j, x[j], :x[j]+1] * (v_sum[j] + v[j, :x[j]+1])
            if w > w_max:
                pi = j
                w_max = w
    return pi


# @nb.njit
def policy(x, i):
    if strategy == 'ospi':
        return ospi(x, i)
    mask = x > 0  # If waiting
    mask[i] = True  # or just arrived
    if strategy == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(mask, x, np.nan))
    elif strategy == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(mask, t - x, np.nan))
    elif strategy == 'sdf_prior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(on_time):
            np.nanargmin(np.where(mask & on_time, t - x, np.nan))
        else:  # FCFS
            np.nanargmax(np.where(mask, x, np.nan))
    elif strategy == 'cmu':
        return np.nanargmax(np.where(mask, cmu, np.nan))


# @nb.njit
def admission(x, s, i, time, arr, n_admit, dep, arr_times, heap, kpi):
    """Assumes that sum(s)<S."""
    pi = policy(x, i)
    if pi < J:  # Take class pi into service, add its departure & new arrival
        kpi[n_admit, :] = time, pi, x[pi]
        n_admit += 1
        s += 1
        hq.heappush(heap, (time + service_times[pi][dep[pi]], pi, 'departure'))
        hq.heappush(heap,
                    (arr_times[pi] + arrival_times[pi][arr[pi]], pi, 'arrival'))
    else:  # Idle
        hq.heappush(heap,
                    (time + 1/gamma, i, 'idle'))
    return heap, kpi, n_admit, s


#@nb.njit
def simulate_multi_class_system(kpi):
    """Simulate a multi-class system."""
    # initialize the system
    time = 0.0
    s = 0
    arr = np.zeros(J, dtype=np.int32)
    arr_times = np.zeros(J, dtype=np.float32)
    n_admit = 0
    dep = np.zeros(J, dtype=np.int32)
    # heap = nb.typed.List.empty_list(heap_type)
    heap = []  # TODO
    # initialize the event list
    for i in range(J):
        hq.heappush(heap, (arrival_times[i][0], i, 'arrival'))
    # run the simulation
    while arr.sum() < N:
        # get next event
        event = hq.heappop(heap)
        time = event[0] if event[0] > time else time
        i = event[1]
        type_event = event[2]
        if type_event in ['arrival', 'idle']:  # arrival of FIL by design
            if type_event == 'arrival':
                arr[i] += 1
                arr_times[i] = event[0]
            if s < S:
                x = time - arr_times
                heap, kpi, n_admit, s = \
                    admission(x, s, i, time, arr, n_admit, dep, arr_times, heap,
                              kpi)
        elif type_event == 'departure':
            dep[i] += 1
            s -= 1  # ensures that sum(s) < S
            x = time - arr_times
            heap, kpi, n_admit, s = \
                admission(x, s, i, time, arr, n_admit, dep, arr_times, heap,
                          kpi)
    return kpi


kpi = simulate_multi_class_system(kpi)
# remove all rows with first column zeros of kpi dataframe
kpi = kpi[~np.all(kpi == 0, axis=1)]
# kpi_df = pd.Dataframe(kpi, columns=['time', 'class', 'wait'])
# kpi_df['g', 'target', 'avg_wait'] = np.nan
# for i in range(J):
#     mask = (kpi[1] == i)
#     kpi[mask, 'reward'] = np.where(kpi[mask, 'wait'] < t[i], r[i], r[i] - c[i])
#     kpi[mask, 'g'] = kpi[mask, 'reward'].cumsum() / kpi[mask, 'reward'].cumsum()


# if __name__ == '__main__':
#     main()
