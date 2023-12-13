"""
This file contains the simulation of the multi-class queueing system.
python simulation.py --job_id 1 --array_id 1 --time 00:05:00 --instance 01 --x 1e5 --method vi

Ospi clocking
python
Sims done: 100000 sims. (N=100000, n_left=100000) Total time: (MM:SS): 00:24,
time per 10,000 iterations: (MM:SS): 00:02
Numba

sdf clocking
python
Sims done: 100000 sims. (N=100000, n_left=100000) Total time: (MM:SS): 00:24,
time per 10,000 iterations: (MM:SS): 00:02
Numba

"""

import heapq as hq
import numba as nb
import numpy as np
import os
import pickle as pkl
# from numba import types as tp
from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import tools

FILEPATH_INSTANCE = 'results/instance_sim_'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_RESULT = 'results/simulation_pickles/result_'

# Debug
args = {'job_id': 1,
        'array_id': 3,
        'time': '0-00:10:00',
        'instance': '09',
        'method': 'not specified',
        'x': 1e5}
args = tools.DotDict(args)
# args = tools.load_args()

inst_nr = int(args.instance) + int((args.array_id - 1)/5)
if inst_nr < 10:
    instance = '0' + str(inst_nr)
else:
    instance = str(inst_nr)

inst = tools.inst_load(FILEPATH_INSTANCE + instance + '.csv')
if args.method in inst['method'].values:
    method_id = (inst['method'] == args.method).idxmax()
else:
    method_id = (args.array_id - 1) % 5
inst = inst.iloc[method_id]
method = inst['method']

# global constants
N = int(args.x) if args.x > 0 else int(1e4)  # arrivals to simulate

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
          seed=inst_nr,  # args.job_id*args.array_id,
          max_time=args.time, max_iter=N, sim='yes')
start_time = env.start_time
max_time = env.max_time
p_xy = env.p_xy
# regret = np.max(r) - r + c
sorted_order = sorted(tuple(zip(c * mu, t, range(J))), reverse=True)
unsorted_order = sorted(zip(sorted_order, range(J)), key=lambda x: x[0][2])
cmu_order = [item[1] for item in unsorted_order]

heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = tools.get_v_app(env)
arrival_times, service_times = tools.generate_times(env, J, lab, mu, N)


# @nb.njit(tp.i8(tp.i4[:], tp.i8, tp.f8[:]))
def ospi(fil, i, x):
    """One-step policy improvement.
    i indicate which class just arrived, i = J if no class arrived.
    """
    x = np.minimum(np.round(x / gamma, 0), D - 1).astype(int)
    # for i in range(J):
    #     x[i] = min(np.round(x[i] / gamma, 0), D - 1)
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
            w += sum(p_xy[j, x[j], :x[j]+1] * (v_sum[j] + v[j, :x[j]+1]))
            if w > w_max:
                pi = j
                w_max = w
    return pi


# @nb.njit(tp.i8(tp.i4[:], tp.i8, tp.f8[:]))
def policy(fil, i, x):
    """Return the class to admit, assumes at least one FIL."""
    if method == 'ospi':
        return ospi(fil, i, x)
    if method == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(fil, x, np.nan))
    elif method == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(fil, t - x, np.nan))
    elif method == 'sdfprior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(fil & on_time):
            return np.nanargmin(np.where(fil & on_time, t - x, np.nan))
        else:  # FCFS
            return np.nanargmin(np.where(fil, x, np.nan))
    elif method == 'cmu':
        return np.nanargmin(np.where(fil, cmu_order, np.nan))


# @nb.njit(tp.Tuple((tp.i4[:], nb.typed.List(heap_type), tp.i8[:], tp.i8, tp.i8))(
#     tp.i4, tp.f4[:], tp.i4,  tp.i4[:], nb.typed.List(heap_type), tp.i8[:],
#     tp.i8, tp.i8[:], tp.i8, tp.i8, tp.i8, tp.f8, tp.f8[:]))
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


# @nb.njit(tp.Tuple((tp.f4[:], tp.i4[:], nb.typed.List, tp.i8[:],
#                    tp.i8, tp.f8))(
#     tp.f4[:], tp.i4[:], nb.typed.List(heap_type), tp.i8[:], tp.i8, tp.i8, tp.f8,
#     tp.i8))
def simulate_multi_class_system(arr_times=np.zeros(J, dtype=np.float32),
                                fil=np.zeros(J, dtype=np.int32),
                                heap=[],  # nb.typed.List.empty_list(heap_type),
                                kpi=np.zeros((N + 1, 3), dtype=np.float64),
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
        if (n_admit % convergence_check) == 0:
            with nb.objmode():
                if (clock() - start_time) > max_time:
                    print(f'Time limit {max_time} reached, stop simulation.')
                    break
    return arr_times, fil, heap, kpi, s, time


def main():
    pickle_file = 'result_' + instance + '_' + method + '.pkl'
    if pickle_file in os.listdir(FILEPATH_PICKLES):
        arr_times, fil, heap, kpi, s, time = pkl.load(open(FILEPATH_PICKLES +
                                                           pickle_file, 'rb'))
        n_done = np.sum(kpi[:, 0] > 0)
        n_left = N - n_done
        if n_left > 2:
            if len(kpi) < N:
                kpi = np.concatenate((kpi, np.zeros((N - len(kpi) + 1, 3))))
            arr_times, fil, heap, kpi, s, time = simulate_multi_class_system(
                arr_times=arr_times,
                fil=fil,
                heap=heap,
                kpi=kpi,
                n_admit=n_done,
                s=s,
                time=time,
                sims=N)
    else:
        arr_times, fil, heap, kpi, s, time = simulate_multi_class_system()
        n_left = N
    if n_left > 0:
        time = clock() - env.start_time
        print(f'Sims done: {np.sum(kpi[:, 0] > 0)} (N={N}, n_left={n_left}). '
              f'Total time: {tools.sec_to_time(time)}, '
              f'time per 10,000 iterations: '
              f'{tools.sec_to_time(time / n_left * 1e4)}.')
        pkl.dump([arr_times, fil, heap, kpi, s, time], open(FILEPATH_PICKLES +
                                                            pickle_file, 'wb'))
    else:
        print(f'Already done {N} sims.')


if __name__ == '__main__':
    main()
