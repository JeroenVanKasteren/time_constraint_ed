"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

python train.py --job_id 1 --array_id 1 --time 0-00:03:00 --instance 01 --method vi

Check how many jobs needed:
import pandas as pd

inst = pd.read_csv('results/instances_01.csv')
print('Solved vi: ' + str(inst['vi_g'].count()) + '\n' +
      'left vi: ' + str(len(inst) - inst['vi_g'].count()) + '\n' +
      'Solved ospi: ' + str(inst['ospi_g'].count()) + '\n' +
      'left ospi: ' + str(len(inst) - inst['ospi_g'].count()) + '\n' +
      'Solved both: ' + str(inst['opt_gap'].count()))

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import argparse
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


def load_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', default='0')  # SULRM_JOBID
    parser.add_argument('--array_id', default='0')  # SLURM_ARRAY_TASK_ID
    parser.add_argument('--time')  # SLURM_TIMELIMIT
    parser.add_argument('--instance', default='01')  # User input
    parser.add_argument('--method')  # User input, vi or ospi
    parser.add_argument('--x', default=0)  # User input
    args = parser.parse_args(raw_args)
    args.job_id = int(args.job_id)
    args.array_id = int(args.array_id)
    args.x = int(args.x)
    return args

# Debug
# args = {'instance': '01', 'method': 'ospi', 'time': '1-00:00:00',
#         'job_id': 1, 'array_id': 1, 'x': 0}
# args = tools.DotDict(args)


def main(raw_args=None):
    args = load_args(raw_args)

    # ---- Problem ---- #
    # seed = args.job_id * args.array_id

    inst = pd.read_csv(FILEPATH_INSTANCE + args.instance + '.csv')
    cols = ['t', 'c', 'r', 'lab', 'mu']
    inst.loc[:, cols] = inst.loc[:, cols].applymap(tools.strip_split)
    inst = inst[pd.isnull(inst[args.method + '_g'])]
    inst[args.method + '_time'] = inst[args.method + '_time'].map(
        lambda x: x if pd.isnull(x) else tools.get_time(x))
    # inst = inst[(inst[args.method + '_time'] < tools.get_time(args.time)) |
    #             pd.isnull(inst[args.method + '_time'])]

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve within', args.time,
              'index:', args.array_id - 1 + args.x)
        exit(0)
    inst = inst.iloc[args.array_id - 1 + args.x]

    env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
              e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
              lab=inst.lab, mu=inst.mu, max_time=args.time,
              convergence_check=10)
    inst[args.method + '_job_id'] = str(args.job_id) + '_' + str(args.array_id)
    inst[args.method + '_time'] = args.time
    inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
                '_' + args.method +
                '_job_' + str(args.job_id) + '_' + str(args.array_id) + '.csv')

    pi_learner = PolicyIteration()
    if args.method == 'vi':
        learner = ValueIteration(env, pi_learner)
        v_file = (FILEPATH_V + 'v_' + args.instance + '_' + str(inst[0])
                  + '_vi.npz')
        if v_file in os.listdir(FILEPATH_V):
            learner.V = np.load(v_file)
        learner.value_iteration(env)
    else:
        learner = OneStepPolicyImprovement(env, pi_learner)
        v_file = (FILEPATH_V + 'v_' + args.instance + '_' + str(inst[0])
                  + '_ospi.npz')
        if v_file in os.listdir(FILEPATH_V):
            learner.V = np.load(v_file)
        learner.one_step_policy_improvement(env)
        learner.get_g(env)

    if learner.converged:
        inst.at[args.method + '_g'] = learner.g
        inst.at[args.method + '_iter'] = learner.iter
        inst.at[args.method + '_time'] = tools.sec_to_time(clock()
                                                           - env.start_time)
        inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
                    '_' + args.method + '_job_' + str(args.job_id) + '_' +
                    str(args.array_id) + '.csv')
    else:
        np.savez(FILEPATH_V + 'v_' + args.instance + '_' + str(inst[0]) + '_'
                 + args.method + '.npz', learner.V)


if __name__ == '__main__':
    main()
