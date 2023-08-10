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
FILEPATH_RESULTS = 'results/results.csv'
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

def main(raw_args=None):
    args = load_args(raw_args)
    # ---- Problem ---- #
    seed = args.job_id * args.array_id
    # f_name = 'Results/' + str(args.id) + '_' + str(args.index) + 'Py.txt'
    args = {'instance': '01', 'method': 'ospi', 'time': '0-00:03:00',
            'job_id': '1', 'array_id': '1', 'x': '0'}  # TODO

    inst = pd.read_csv(FILEPATH_instances + args.instance + '.csv',
                       na_values=np.nan)
    inst = inst[np.isnan(inst[args.method + '_g'])]
    inst[args.method + '_time'].map(Env.get_time)

    inst = inst[inst[args.method + '_time'] < Env.get_time(max_time)]

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve, index:', args.array_id - 1 + args.x )
        exit(0)
    inst = inst.iloc[args.array_id - 1 + args.x]

    if not np.isnan(inst[args.method + '_g']):
        print('Already solved')
        exit(0)

    env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
              e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
              lab=inst.lab, mu=inst.mu, max_time=args.time)

    if np.isnan(inst[args.method+'_time']) | (inst[args.method+'_time'] > args.time):
        print('Time not sufficient, time:', inst[args.method+'_time'], 'max time:', args.time)
        exit(0)

    if inst
    pi_learner = PolicyIteration()
    # ---- Value Iteration ---- #
    vi_learner = ValueIteration(env, pi_learner)
    vi_learner.value_iteration(env)

    # ---- One Step Policy Improvement ---- #
    ospi_learner = OneStepPolicyImprovement(env, pi_learner)
    ospi_learner.get_g(env, V=vi_learner.V)

    result = [instance_id, args.job_id, args.array_id,
              datetime.today().strftime('%Y-%m-%d'),
              seed, env.J, env.S, env.D, env.size, env.size_i, env.gamma, env.e,
              env.t, env.c, env.r, env.lab, env.mu, env.load, env.cap_prob,
              env.weighted_cap_prob, vi_learner.converged,
              ospi_learner.converged, (clock() - env.start_time),
              vi_learner.g, ospi_learner.g,
              abs(vi_learner.g - ospi_learner.g) / vi_learner.g]

    vi_learner.get_policy(env)
    np.savez('Results/policy_' + args.id + '.npz',
             vi_learner.Pi, ospi_learner.Pi,
             vi_learner.V, ospi_learner.V_app)
    # np.load('Results/policy_SLURM_ARRAY_TASK_ID.npz')

    # Was it solved?
    # TODO: save results to csv
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
