"""
Iteratively solves different environments.

Setup:
N classes, N=2,3,4 (SIMS := # experiments each)

python Train.py --id=1 --index=1 --J=2 --gamma=25 --policy=False --time=0-00:03:00

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
Created on 19-3-2020.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from time import perf_counter as clock
from Env_and_Learners import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement

FILEPATH_instances = 'results/instances_'
FILEPATH_RESULTS = 'results/result_'
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

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main(raw_args=None):
    args = load_args(raw_args)

    args = {'instance': '01', 'method': 'ospi', 'time': '0-00:03:00',
            'job_id': 1, 'array_id': 1, 'x': 0}  # TODO
    args = dotdict(args)
    # ---- Problem ---- #
    seed = args.job_id * args.array_id
    # f_name = 'Results/' + str(args.id) + '_' + str(args.index) + 'Py.txt'

    inst = pd.read_csv(FILEPATH_instances + args.instance + '.csv')
    inst = inst[np.isnan(inst[args.method + '_g'])]
    inst[args.method + '_time'] = inst[args.method + '_time'].map(Env.get_time)
    inst = inst[inst[args.method + '_time'] < Env.get_time(args.time)]

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve within', args.time,
              'index:', args.array_id - 1 + args.x )
        exit(0)
    inst = inst.iloc[args.array_id - 1 + args.x]

    env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
              e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
              lab=inst.lab, mu=inst.mu, max_time=args.time)

    pi_learner = PolicyIteration()
    if args.method == 'vi':
        learner = ValueIteration(env, pi_learner)
        learner.value_iteration(env)
    else:
        learner = OneStepPolicyImprovement(env, pi_learner)
        learner.one_step_policy_improvement(env)
        learner.get_g(env)

    # vi_learner.get_policy(env)
    # np.savez('Results/policy_' + args.id + '.npz',
    #          vi_learner.Pi, ospi_learner.Pi,
    #          vi_learner.V, ospi_learner.V_app)
    # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')

    # Was it solved?
    # TODO: save results to csv
    inst.at[args.method + '_g'] = learner.g
    inst.at[args.method + '_iter'] = learner.iter
    inst.at[args.method + '_time'] = learner.time
    inst.to_csv(FILEPATH_RESULTS + args.instance + '_' + inst[0] +
                '_' + args.method + '.csv')
    # import csv
    #
    # myDic = {"a": 1, "b": 2, "c": 15}
    # with open('myFile.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['word', 'count']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #     writer.writeheader()
    #     for key in myDic:
    #         writer.writerow({'word': key, 'count': myDic[key]})
    #
    # Path(FILEPATH).touch()
    # with open(FILEPATH, 'a') as f:  # a = append
    #     f.write(','.join(map(str, result)) + '\n')
    #     f.flush()
    #     os.fsync()

if __name__ == '__main__':
    main()
