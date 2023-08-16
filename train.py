"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

python train.py --job_id 1 --array_id 1 --time 0-00:03:00 --instance 01 --method vi

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import argparse
import pandas as pd
from time import perf_counter as clock
from utils import tools, TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement

FILEPATH_INSTANCE = 'results/instances_'
FILEPATH_RESULT = 'results/result_'
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
    args.x = float(args.x)
    return args

# Debug
# args = {'instance': '01', 'method': 'ospi', 'time': '0-00:03:00',
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
    inst = inst[(inst[args.method + '_time'] < tools.get_time(args.time)) |
                pd.isnull(inst[args.method + '_time'])]

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve within', args.time,
              'index:', args.array_id - 1 + args.x )
        exit(0)
    inst = inst.loc[args.array_id - 1 + args.x]

    env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
              e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
              lab=inst.lab, mu=inst.mu, max_time=args.time)
    inst[args.method + '_job_id'] = str(args.job_id) + '_' + str(args.array_id)
    inst[args.method + '_time'] = args.time
    inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
                '_' + args.method +
                '_job_' + str(args.job_id) + '_' + str(args.array_id) + '.csv')

    pi_learner = PolicyIteration()
    if args.method == 'vi':
        learner = ValueIteration(env, pi_learner)
        learner.value_iteration(env)
    else:
        learner = OneStepPolicyImprovement(env, pi_learner)
        learner.one_step_policy_improvement(env)
        learner.get_g(env)

    # save matrices
    # vi_learner.get_policy(env)
    # np.savez('Results/policy_' + args.id + '.npz',
    #          vi_learner.Pi, ospi_learner.Pi,
    #          vi_learner.V, ospi_learner.V_app)
    # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')

    if learner.converged:
        inst.at[args.method + '_g'] = learner.g
        inst.at[args.method + '_iter'] = learner.iter
        inst.at[args.method + '_time'] = clock() - env.start_time
        inst.to_csv(FILEPATH_RESULT + args.instance + '_' + str(inst[0]) +
                    '_' + args.method + '_job_' + str(args.job_id) + '_' +
                    str(args.array_id) + '.csv')


if __name__ == '__main__':
    main()
