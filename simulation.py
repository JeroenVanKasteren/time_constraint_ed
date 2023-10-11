"""
This file contains the simulation of the multi-class queueing system.
"""

import heapq as hq
import numba as nb
import numpy as np
import os
import pickle as pkl
from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import tools

FILEPATH_INSTANCE = 'results/instance_sim_'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_RESULT = 'results/simulation_pickles/result_'

# Debug
args = {'job_id': 1,
        'array_id': 2,
        'time': '0-00:30:00',
        'instance': '01',
        'method': 'not_specified',
        'x': 1e4}
args = tools.DotDict(args)
# args = tools.load_args()  # TODO

inst = tools.inst_load(FILEPATH_INSTANCE + args.instance + '.csv')
if args.method in inst['method']:
    method_id = inst['method'].lt(args.method).idxmax()
else:
    method_id = args.array_id
inst = inst.iloc[method_id]
method = inst['method']

# global constants
N = int(args.x) if args.x > 0 else int(1e3)  # arrivals to simulate
# Moreover, sum up N when doing multiple runs (continuing runs).
convergence_check = 1e4
J, S, gamma, D, t, c, r, mu, lab, load, imbalance = (inst.J,
                                                     inst.S,
                                                     inst.gamma,
                                                     inst.D,
                                                     inst.t,
                                                     inst.c,
                                                     inst.r,
                                                     inst.mu,
                                                     inst.lab,
                                                     inst.load,
                                                     inst.imbalance)

env = Env(J=J, S=S, D=D, gamma=gamma, t=t, c=c, r=r, mu=mu, lab=lab,
          seed=args.job_id*args.array_id,
          max_time=args.time, max_iter=args.max_iter, sim=1)
max_time = env.max_time
p_xy = env.p_xy
# regret = np.max(r) - r + c
cmu = c * mu

heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = tools.get_v_app(env)
arrival_times, service_times = tools.generate_times(env, J, lab, mu, N)


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
    if method == 'ospi':
        return ospi(fil, i, x)
    if method == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(fil, x, np.nan))
    elif method == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(fil, t - x, np.nan))
    elif method == 'sdf_prior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(on_time):
            np.nanargmin(np.where(fil & on_time, t - x, np.nan))
        else:  # FCFS
            np.nanargmax(np.where(fil, x, np.nan))
    elif method == 'cmu':
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


# @nb.njit
def simulate_multi_class_system(arr_times=np.zeros(J, dtype=np.float32),
                                fil=np.zeros(J, dtype=np.int32),
                                heap=[],  # heap = nb.typed.List.empty_list(heap_type)
                                kpi=np.zeros((N + 1, 3)),
                                n_admit=0,
                                s=0,
                                time=0.0,
                                sims=N):
    """Simulate a multi-class system."""
    arr = np.zeros(J, dtype=np.int32)
    dep = np.zeros(J, dtype=np.int32)
    if len(heap) == 0:
        for i in range(J):  # initialize the event list
            hq.heappush(heap, (arrival_times[i][0], i, 'arrival'))
    while n_admit < sims:
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
    return arr_times, fil, heap, kpi, s, time


def main():
    pickle_file = (FILEPATH_RESULT + args.instance + '_' + method + '.pkl')
    if pickle_file in os.listdir(FILEPATH_PICKLES):
        arr_times, fil, heap, kpi, s, time = pkl.load(open(pickle_file, 'rb'))
        m = np.sum(kpi[:, 0] == 0)  # simulations left
        if m < 2:
            kpi = np.concat((kpi, np.zeros((N + 1, 3))))
            n, m = N, 0
        else:
            n, m = m, len(kpi) - m
        kpi = simulate_multi_class_system(arr_times=arr_times,
                                          fil=fil,
                                          heap=heap,
                                          kpi=kpi,
                                          n_admit=m,
                                          s=s,
                                          time=time,
                                          sims=n)
    else:
        arr_times, fil, heap, kpi, s, time = simulate_multi_class_system()
    time = clock() - env.start_time
    print(f'Total time: {tools.sec_to_time(time)}, '
          f'time per iteration: {tools.sec_to_time(time / N)}')
    pkl.dump([arr_times, fil, heap, kpi, s, time], open(pickle_file, 'wb'))


if __name__ == '__main__':
    main()
