"""
This file contains the simulation of the multi-class queueing system.

Convert to a class when it works.
"""

import argparse
import heapq as hq
import numba as nb
import numpy as np
# import pandas as pd
from time import perf_counter as clock
from utils import TimeConstraintEDs as Env
from utils import OneStepPolicyImprovement as Ospi

np.set_printoptions(precision=4, linewidth=150, suppress=True)

N = 1000  # arrivals to simulate
BATCH_SIZE = 100  # batch size for KPI

env = Env(J=2, S=4, gamma=10, D=50, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=10, print_modulo=100)

heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)

J = env.J
S = env.S
gamma = env.gamma
t = env.t
c = env.c
r = env.r
lab = env.lab
mu = env.mu
p_xy = env.p_xy
regret = np.max(r) - r + c

arrival_times = np.empty((J, N + 1), dtype=np.float32)  # +1 for last arrival
service_times = np.empty((J, N + 1), dtype=np.float32)
for i in range(J):
    arrival_times[i, :] = env.rng.exponential(1 / lab[i], N + 1)
    service_times[i, :] = env.rng.exponential(1 / mu[i], N + 1)

policy = 'fcfs'  # 'fcfs' 'sdf' 'sdf_prior' 'cmu' 'ospi'


@nb.njit
def update_mean(mean, x, n):
    """Welford's method to update the mean."""
    return mean + (x - mean) / n  # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n


@nb.njit
def ospi(env, pi_learner):
    Ospi.get_v_app_i(env, i)

    pi = 0
    # x
    # s
    # state = (i * d_i['sizes_i'][0] + np.sum(x * sizes_x + s * sizes_s))
    # j is the class to admit
    # i indicates which class just arrived
    # i = J if no class arrived
    for j in range(J):
        if (x[j] > 0) or (j == i):
            w = r[j] - c[j] if x[j] > gamma * t[j] else r[j]
            i_not_admitted = 0
            if (i < J) and (i != j):
                i_not_admitted = sizes_x_n[i]
            for y in range(x[j] + 1):
                next_state = (np.sum(
                    x * sizes_x_n + s * sizes_s_n)
                              - (x[j] - y) * sizes_x_n[j]
                              + i_not_admitted
                              + sizes_s_n[j])
                w += P_xy[j, x[j], y] * V[next_state]
            if w > W[state]:
                W[state] = w


@nb.njit
def policy(arr_times, time, x, s):
    mask = x > 0  # If waiting
    mask[i] = True  # or just arrived
    if policy == 'fcfs':  # argmax(x)
        return np.nanargmax(np.where(mask, x, np.nan))
    elif policy == 'sdf':  # argmin(t - x)
        return np.nanargmin(np.where(mask, t - x, np.nan))
    elif policy == 'sdf_prior':
        y = t - x  # Time till deadline
        on_time = y >= 0
        if np.any(on_time):
            np.nanargmin(np.where(mask & on_time, t - x, np.nan))
        else:  # FCFS
            np.nanargmax(np.where(mask, x, np.nan))
    elif policy == 'cmu':
        return np.argmax(regret * x * lab * mu)
    elif policy == 'ospi':
        return ospi(x, s) - 1  # TODO -1?


@nb.njit
def admission(avg_wait, heap, total_reward, arr, arr_times, dep, x, s, time):
    """Assumes that sum(s)<S."""
    pi = policy(arr_times, time, x, s)
    if pi < J:  # add departure & arrival of class pi
        hq.heappush(heap, (time + service_times[pi, dep[pi]], pi, 'departure'))
        total_reward += r[pi] - c[pi] if x[pi] > gamma * t[pi] else r[pi]
        avg_wait[pi] = update_mean(avg_wait[pi], x[pi], arr[pi])
        hq.heappush(heap,
                    (arr_times[pi] + arrival_times[pi, arr[pi]], pi, 'arrival'))
    return avg_wait, heap, total_reward


# @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
#                   DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
#          parallel=True, error_model='numpy')
def simulate_multi_class_system(N, arrival_times, service_times):
    # initialize the system
    time = 0.0
    s = np.zeros(J+1, dtype=np.int32)
    total_reward = np.zeros(J+1)
    arr = np.ones(J, dtype=np.int32)
    dep = np.zeros(J, dtype=np.int32)
    avg_wait = np.zeros(J, dtype=np.float32)
    arr_times = np.zeros(J, dtype=np.float32)
    heap = nb.typed.List.empty_list(heap_type)

    # initialize the event list
    for i in range(J):
        hq.heappush(heap, (arrival_times[i, 0], i, 'arrival'))

    # run the simulation
    while arr.sum() < N:
        # get next event
        event = hq.heappop(heap)
        time = event[0] if event[0] < time else time
        i = event[1]
        type_event = event[2]

        if type_event == 'arrival':  # arrival of FIL by design
            arr[i] += 1
            arr_times[i] = event[0]
            if np.sum(s) < S:
                x = time - arr_times
                avg_wait, heap, total_reward = admission(avg_wait, heap,
                                                         total_reward, arr,
                                                         arr_times, dep, x, s,
                                                         time)
        else:  # type_event == 'departure':
            avg_wait, heap, total_reward = admission(avg_wait, heap,
                                                     total_reward, arr,
                                                     arr_times, dep, x, s,
                                                     time)


# def load_args(raw_args=None):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--job_id', default='0')  # SULRM_JOBID
#     parser.add_argument('--array_id', default='0')  # SLURM_ARRAY_TASK_ID
#     parser.add_argument('--time')  # SLURM_TIMELIMIT
#     parser.add_argument('--instance', default='01')  # User input
#     parser.add_argument('--method')  # User input, vi or ospi
#     parser.add_argument('--x', default=0)  # User input
#     args = parser.parse_args(raw_args)
#     args.job_id = int(args.job_id)
#     args.array_id = int(args.array_id)
#     args.x = float(args.x)
#     return args
#
# # Debug
# # args = {'instance': '01', 'method': 'ospi', 'time': '0-00:03:00',
# #         'job_id': 1, 'array_id': 1, 'x': 0}
# # args = tools.DotDict(args)
#
#
# def main(raw_args=None):
#     args = load_args(raw_args)
#
#     # ---- Problem ---- #
#     # seed = args.job_id * args.array_id
#
#     inst = pd.read_csv(FILEPATH_INSTANCE + args.instance + '.csv')
#     cols = ['t', 'c', 'r', 'lab', 'mu']
#     inst.loc[:, cols] = inst.loc[:, cols].applymap(tools.strip_split)
#     inst = inst[pd.isnull(inst[args.method + '_g'])]
#     inst[args.method + '_time'] = inst[args.method + '_time'].map(
#         lambda x: x if pd.isnull(x) else tools.get_time(x))
#     inst = inst[(inst[args.method + '_time'] < tools.get_time(args.time)) |
#                 pd.isnull(inst[args.method + '_time'])]
#
#     if args.array_id - 1 + args.x >= len(inst):
#         print('No more instances to solve within', args.time,
#               'index:', args.array_id - 1 + args.x )
#         exit(0)
#     inst = inst.loc[args.array_id - 1 + args.x]
#
#     env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
#               e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
#               lab=inst.lab, mu=inst.mu, max_time=args.time)
#     inst[args.method + '_job_id'] = str(args.job_id) + '_' + str(args.array_id)
#     inst[args.method + '_time'] = args.time
#     inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
#                 '_' + args.method +
#                 '_job_' + str(args.job_id) + '_' + str(args.array_id) + '.csv')
#
#     pi_learner = PolicyIteration()
#     if args.method == 'vi':
#         learner = ValueIteration(env, pi_learner)
#         learner.value_iteration(env)
#     else:
#         learner = OneStepPolicyImprovement(env, pi_learner)
#         learner.one_step_policy_improvement(env)
#         learner.get_g(env)
#
#     # save g trajectory
#     # vi_learner.get_policy(env)
#     # np.savez('Results/policy_' + args.id + '.npz',
#     #          vi_learner.Pi, ospi_learner.Pi,
#     #          vi_learner.V, ospi_learner.V_app)
#     # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')
#
#     if learner.converged:
#         inst.at[args.method + '_g'] = learner.g
#         inst.at[args.method + '_g_var'] = learner.N
#         inst.at[args.method + '_time'] = tools.sec_to_time(clock()
#                                                            - env.start_time)
#         inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
#                     '_' + args.method + '_job_' + str(args.job_id) + '_' +
#                     str(args.array_id) + '.csv')
#
#
# if __name__ == '__main__':
#     main()
#

