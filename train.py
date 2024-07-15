"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

python train.py --job_id 1 --array_id 1 --time 0-00:03:00 --instance 02 --method vi

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import numpy as np
import os
import pandas as pd
from time import perf_counter as clock
from utils import tools, TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement

FILEPATH_INSTANCE = 'results/instances_'
FILEPATH_RESULT = 'results/result_'
FILEPATH_V = 'results/value_functions/'
MAX_TARGET_PROB = 0.9
# Debug
args = {'instance': 'J1', 'method': 'pi', 'time': '0-00:05:00',
        'job_id': 1, 'array_id': 2, 'x': 0}
args = tools.DotDict(args)


def main(raw_args=None):
    # args = tools.load_args(raw_args)

    inst = pd.read_csv(FILEPATH_INSTANCE + args.instance + '.csv')
    cols = ['t', 'c', 'r', 'lab', 'mu']
    inst.loc[:, cols] = inst.loc[:, cols].map(tools.strip_split)
    inst = inst[pd.isnull(inst[args.method + '_g'])]

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve within', args.time,
              'index:', args.array_id - 1 + args.x, flush=True)
        exit(0)
    inst = inst.iloc[args.array_id - 1 + args.x]

    env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
              e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
              lab=inst.lab, mu=inst.mu, max_time=args.time,
              convergence_check=10)
    inst[args.method + '_job_id'] = str(args.job_id) + '_' + str(args.array_id)
    inst[args.method + '_job_id'] = str(args.job_id) + '_' + str(args.array_id)
    inst[args.method + '_time'] = args.time
    inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst.iloc[0]) +
                '_' + args.method +
                '_job_' + str(args.job_id) + '_' + str(args.array_id) + '.csv')

    pi_learner = PolicyIteration()
    if args.method == 'vi':
        learner = ValueIteration(env, pi_learner)
        v_file = ('v_' + args.instance + '_' + str(inst.iloc[0]) + '_vi.npz')
        if v_file in os.listdir(FILEPATH_V):
            print('Loading V from file', flush=True)
            learner.V = np.load(FILEPATH_V + v_file)['arr_0']
        learner.value_iteration(env)
        pi_file = ('pi_' + args.instance + '_' + str(inst.iloc[0]) + '_vi.npz')
        if learner.converged and (pi_file not in os.listdir(FILEPATH_V)):
            learner.get_policy(env)
    elif args.method == 'ospi':
        learner = OneStepPolicyImprovement(env, pi_learner)
        pi_file = ('pi_' + args.instance + '_' + str(inst.iloc[0]) + '_ospi.npz')
        if pi_file in os.listdir(FILEPATH_V):
            print('Loading Pi from file', flush=True)
            learner.Pi = np.load(FILEPATH_V + pi_file)['arr_0']
        else:
            learner.one_step_policy_improvement(env)
        v_file = ('v_' + args.instance + '_' + str(inst.iloc[0]) + '_ospi.npz')
        if v_file in os.listdir(FILEPATH_V):
            print('Loading V from file', flush=True)
            learner.V = np.load(FILEPATH_V + v_file)['arr_0']
            learner.get_g(env, learner.V)
        else:
            learner.get_g(env, learner.V_app)
    elif args.method in ['sdf', 'fcfs',
                         'cmu_t_min', 'cmu_t_max', 'l_max', 'l_min']:
        learner = OneStepPolicyImprovement(env, pi_learner)
        order = tools.fixed_order(env, args.method)  # None for not fixed order
        learner.Pi = pi_learner.init_pi(env, args.method, order)
        v_file = ('v_' + args.instance + '_' + str(inst.iloc[0]) + '_' +
                  args.method + '.npz')
        if v_file in os.listdir(FILEPATH_V):
            print('Loading V from file', flush=True)
            learner.V = np.load(FILEPATH_V + v_file)['arr_0']
        else:
            learner.V = np.zeros(env.dim, dtype=np.float32)
        learner.get_g(env, learner.V)
    elif args.method == 'pi':
        max_pi_iter = 4
        learner = pi_learner
        pi_file = ('pi_' + args.instance + '_' + str(inst.iloc[0]) + '_pi.npz')
        v_file = ('v_' + args.instance + '_' + str(inst.iloc[0]) + '_pi.npz')
        g_file = ('g_' + args.instance + '_' + str(inst.iloc[0]) + '_pi.npz')
        if ((pi_file in os.listdir(FILEPATH_V)) &
                (v_file in os.listdir(FILEPATH_V))):
            print('Loading pi & v from file', flush=True)
            Pi = np.load(FILEPATH_V + pi_file)['arr_0']
            V = np.load(FILEPATH_V + v_file)['arr_0']
            g_mem = np.load(FILEPATH_V + g_file)['arr_0']
            g_mem = learner.policy_iteration(env, g_mem=g_mem, Pi=Pi, V=V, max_pi_iter=max_pi_iter)
        else:
            g_mem = learner.policy_iteration(env, max_pi_iter=max_pi_iter)
        np.savez(FILEPATH_V + 'g_' + args.instance + '_' + str(inst.iloc[0]) + '_'
                 + args.method + '.npz', g_mem)
    else:
        print('Method not recognized', flush=True)
        exit(0)

    if args.method != 'pi':
        if learner.g != 0:
            inst.at[args.method + '_g_tmp'] = learner.g
        if learner.converged:
            inst.at[args.method + '_g'] = learner.g

        inst.at[args.method + '_time'] = \
            tools.sec_to_time(clock() - env.start_time +
                              tools.get_time(inst.at[args.method + '_time']))
        inst.at[args.method + '_iter'] += learner.iter
        inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst.iloc[0]) +
                    '_' + args.method + '_job_' + str(args.job_id) + '_' +
                    str(args.array_id) + '.csv')

    if learner.V is not None:
        np.savez(FILEPATH_V + 'v_' + args.instance + '_' + str(inst.iloc[0]) + '_'
                 + args.method + '.npz', learner.V)
    if learner.Pi is not None:
        np.savez(FILEPATH_V + 'pi_' + args.instance + '_' + str(inst.iloc[0]) + '_'
                 + args.method + '.npz', learner.Pi)


if __name__ == '__main__':
    main()
