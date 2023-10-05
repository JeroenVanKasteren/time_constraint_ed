"""
This file contains the simulation of the multi-class queueing system.
"""

# import argparse
import heapq as hq
import numba as nb
import numpy as np
import pandas as pd
from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import tools

FILEPATH_INSTANCE = 'results/instance_sim_'
FILEPATH_RESULT = 'results/result_'
FILEPATH_PICKLES = 'results/value_functions/'

# global constants
N = 1e3  # arrivals to simulate determine when starting the running
# Moreover, sum up N when doing multiple runs (continuing runs).
start_K = 1e3
batch_T = 1e4
batch_size = 1000  # batch size for KPI
convergence_check = 1e4
strategy = 'fcfs'  # 'fcfs' 'sdf' 'sdf_prior' 'cmu' 'ospi'
instance = '01'

inst = tools.inst_load(FILEPATH_INSTANCE + instance + '.csv')

J = inst.J
S = inst.S
gamma = inst.J
D = inst.D
t = inst.t
c = inst.c
r = inst.r
mu = inst.mu
load = inst.load
imbalance = inst.imbalance

env = Env(J=J, S=S, D=D, gamma=gamma, t=t, c=c, r=r, mu=mu, load=load,
          imbalance=imbalance)
# lab=lab, e=0.1, max_time=args.time)
lab = env.lab
p_xy = env.p_xy
regret = np.max(r) - r + c
cmu = c * mu
# start_time = env.start_time
# max_time = env.max_time
# broke = False

heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = tools.get_v_app(env)
arrival_times, service_times = tools.generate_times(env, J, lab, mu, N)
kpi_np = np.zeros((N+1, 3))  # time, class, waited


# @nb.njit
def ospi(fil, i, x):
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
        if fil[j]:
            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
            w += p_xy[j, x[j], :x[j]+1] * (v_sum[j] + v[j, :x[j]+1])
            if w > w_max:
                pi = j
                w_max = w
    return pi


# @nb.njit
def policy(fil, i, x):
    """Return the class to admit, assumes at least one FIL."""
    if strategy == 'ospi':
        return ospi(fil, i, x)
    if strategy == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(fil, x, np.nan))
    elif strategy == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(fil, t - x, np.nan))
    elif strategy == 'sdf_prior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(on_time):
            np.nanargmin(np.where(fil & on_time, t - x, np.nan))
        else:  # FCFS
            np.nanargmax(np.where(fil, x, np.nan))
    elif strategy == 'cmu':
        return np.nanargmax(np.where(fil, cmu, np.nan))


# @nb.njit
def admission(arr, arr_times, dep, fil, heap, i, kpi, n_admit, s, time, x):
    """Assumes that sum(s)<S."""
    pi = policy(fil, i, x)
    if pi < J:  # Take class pi into service, add its departure & new arrival
        kpi[n_admit, :] = time, pi, x[pi]
        n_admit += 1
        s += 1
        fil[pi] = 0
        hq.heappush(heap, (time + service_times[pi][dep[pi]], pi, 'departure'))
        hq.heappush(heap,
                    (arr_times[pi] + arrival_times[pi][arr[pi]], pi, 'arrival'))
    else:  # Idle
        hq.heappush(heap, (time + 1/gamma, i, 'idle'))
    return fil, heap, kpi, n_admit, s


#@nb.njit
def simulate_multi_class_system(kpi):
    """Simulate a multi-class system."""
    time = 0.0  # initialize the system
    s = 0
    fil = np.zeros(J, dtype=np.int32)
    arr = np.zeros(J, dtype=np.int32)
    arr_times = np.zeros(J, dtype=np.float32)
    n_admit = 0
    dep = np.zeros(J, dtype=np.int32)
    # heap = nb.typed.List.empty_list(heap_type)
    heap = []  # TODO
    for i in range(J):  # initialize the event list
        hq.heappush(heap, (arrival_times[i][0], i, 'arrival'))
    while n_admit < N:
        event = hq.heappop(heap)  # get next event
        time = event[0] if event[0] > time else time
        i = event[1]
        type_event = event[2]
        if type_event in ['arrival', 'idle']:  # arrival of FIL by design
            if type_event == 'arrival':
                arr[i] += 1
                fil[i] = 1
                arr_times[i] = event[0]
            if s < S:
                x = np.where(fil, time - arr_times, 0)
                fil, heap, kpi, n_admit, s = admission(arr, arr_times, dep, fil,
                                                       heap, i, kpi, n_admit, s,
                                                       time, x)
        elif type_event == 'departure':
            dep[i] += 1
            s -= 1  # ensures that sum(s) < S
            if sum(fil) > 0:
                x = np.where(fil, time - arr_times, 0)
                fil, heap, kpi, n_admit, s = admission(arr, arr_times, dep, fil,
                                                       heap, i, kpi, n_admit, s,
                                                       time, x)
        # if (n_admit % convergence_check) == 0:
        #     with nb.objmode():
        #         if (clock() - start_time) > max_time:
        #             broke = True
        #             break
    return kpi


kpi_np = simulate_multi_class_system(kpi_np)


# print(tools.sec_to_time(clock() - env.start_time))
# if __name__ == '__main__':
#     main()
