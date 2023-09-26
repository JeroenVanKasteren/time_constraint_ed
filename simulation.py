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
batch_size = 100  # batch size for KPI
policy = 'fcfs'  # 'fcfs' 'sdf' 'sdf_prior' 'cmu' 'ospi'

env = Env(J=3, S=4, gamma=10, D=50, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=10, print_modulo=100)

J = env.J
S = env.S
gamma = env.gamma
D = env.D
t = env.t
c = env.c
r = env.r
lab = env.lab
mu = env.mu
p_xy = env.p_xy
regret = np.max(r) - r + c
cmu = c * mu

# @nb.njit
# def update_mean(mean, x, n):
#    """Welford's method to update the mean."""
#    return mean + (x - mean) / n  # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n


def generate_times(J, N, lab, mu):
    """Generate exponential arrival and service times."""
    arrival_times = np.empty((J, N + 1), dtype=np.float32)  # +1, last arrival
    service_times = np.empty((J, N + 1), dtype=np.float32)
    for i in range(J):
        arrival_times[i, :] = env.rng.exponential(1 / lab[i], N + 1)
        service_times[i, :] = env.rng.exponential(1 / mu[i], N + 1)
    return arrival_times, service_times


def get_v_app(env):
    """Get the approximate value function for a given state."""
    v = np.zeros((env.J, env.D + 1))
    for i in range(env.J):
        v[i,] = Ospi.get_v_app_i(env, i)
    return v


heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)
eye = np.eye(J, dtype=int)
v = get_v_app(env)
arrival_times, service_times = generate_times(J, N, lab, mu)
kpi = np.zeros((N, 5))  # time, class, event, target_reached, waited


@nb.njit
def ospi(x, i):
    """One-step policy improvement.
    i indicates which class just arrived, i = J if no class arrived.
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


@nb.njit
def policy(x, i):
    if policy == 'ospi':
        return ospi(x, i)
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
        return np.nanargmax(np.where(mask, cmu, np.nan))


@nb.njit
def admission(x, s, i, time, arr, dep, arr_times, heap, avg_wait, tot_reward):
    """Assumes that sum(s)<S."""
    pi = policy(x, i)
    if pi < J:  # Take class pi into service, add its departure & new arrival
        s[pi] += 1
        hq.heappush(heap, (time + service_times[pi, dep[pi]], pi, 'departure'))
        tot_reward += r[pi] - c[pi] if x[pi] > gamma * t[pi] else r[pi]
        avg_wait[pi] = update_mean(avg_wait[pi], x[pi], arr[pi])
        hq.heappush(heap,
                    (arr_times[pi] + arrival_times[pi, arr[pi]], pi, 'arrival'))
    return avg_wait, heap, tot_reward, s


@nb.njit
def simulate_multi_class_system(kpi):
    """Simulate a multi-class system."""
    # initialize the system
    time = 0.0
    s = np.zeros(J+1, dtype=np.int32)
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
                avg_wait, heap, tot_reward, s = \
                    admission(x, s, i, time, arr, dep, arr_times, heap,
                              avg_wait, tot_reward)
        else:  # type_event == 'departure':
            dep[i] += 1
            s[i] -= 1  # ensures that sum(s) < S
            x = time - arr_times
            avg_wait, heap, tot_reward, s = \
                admission(x, s, J, time, arr, dep, arr_times, heap, avg_wait,
                          tot_reward)
    return tot_reward, arr, dep, avg_wait, arr_times


simulate_multi_class_system
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

