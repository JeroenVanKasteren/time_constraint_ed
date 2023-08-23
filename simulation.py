"""
This file contains the simulation of the multi-class queueing system.

Convert to a class when it works.
"""

import heapq as hq
import numba as nb
import numpy as np
from utils import TimeConstraintEDs as Env
from time import perf_counter as clock

np.set_printoptions(precision=4, linewidth=150, suppress=True)

N = 1000  # arrivals to simulate
BATCH_SIZE = 100  # batch size for KPI

env = Env(J=2, S=4, gamma=10, D=50, P=1e3, e=1e-5, seed=42,
          max_time='0-00:10:30', convergence_check=10, print_modulo=100)

rng = np.random.default_rng()  # add seed
arrival_times = np.empty((env.J, N), dtype=np.float32)
service_times = np.empty((env.J, N), dtype=np.float32)
for i in range(env.J):
    arrival_times[i, :] = env.rng.exponential(1 / env.lab[i], N)
    service_times[i, :] = env.rng.exponential(1 / env.mu[i], N)

heap_type = nb.typeof((0.0, 0, 'event'))  # (time, class, event)

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

# ensure this as global variable
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

# @nb.njit(tp.f4[:](tp.f4[:], tp.f4[:], tp.i8, tp.i8, tp.f8,
#                   DICT_TYPE_I1, DICT_TYPE_I2, DICT_TYPE_F1, tp.f8[:, :, :]),
#          parallel=True, error_model='numpy')
def simulate_multi_class_system(J, S, lab_i, mu_i, r_i, c_i, t_i, N,
                                arrival_times, service_times):
    # initialize the system
    state = np.zeros(J+1, dtype=np.int32)
    total_reward = np.zeros(J+1)
    total_cost = np.zeros(J+1)
    time = 0
    arr_per_class = np.ones(J, dtype=np.int32)
    ser_per_class = np.zeros(J, dtype=np.int32)
    exp_wait_per_class = np.zeros(J, dtype=np.float32)
    wait_per_class = np.zeros(J, dtype=np.float32)
    heap = nb.typed.List.empty_list(heap_type)

    # initialize the event list
    for i in range(J):
        hq.heappush(heap, (arrival_times[i, 0], i, 'arrival'))

    # run the simulation
    while arr_per_class.sum() < N:
        # get next event
        event = hq.heappop(heap)
        time = event[0]
        j = event[1]
        type_event = event[2]

        if type_event == 'arrival':
            arr_per_class[j] += 1
            # https://stackoverflow.com/questions/12636613/how-to-calculate-moving-average-without-keeping-the-count-and-data-total
            # Welford's method
            exp_wait_per_class[j] += wait_per_class[j] * state[j]
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

        # add new arrival
        hq.heappush(heap,
                    (time + arrival_times[i, arr_per_class[i]], i, 'arrival'))
        # add service completion
        hq.heappush(heap,
                    (time + service_times[i, ser_per_class[i]], i, 'departure'))

        # check if any service initiation events can occur
        if np.sum(state) < S:
            prob = r_i / np.sum(r_i)
            i = np.random.choice(J, p=prob)
            if state[i] > 0:
                state[i] -= 1
                total_reward[i] += r_i[i]
                events.append((t, i, 'departure'))






#
#
#
#
#
#
# import argparse
# import pandas as pd
# from time import perf_counter as clock
# from utils import tools, TimeConstraintEDs as Env, PolicyIteration, \
#     ValueIteration, OneStepPolicyImprovement
#
# FILEPATH_INSTANCE = 'results/instances_'
# FILEPATH_RESULT = 'results/result_'
# MAX_TARGET_PROB = 0.9
#
#
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

