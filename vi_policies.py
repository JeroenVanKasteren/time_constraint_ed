"""
Sandbox
"""

import numpy as np
import os
import pandas as pd
from utils import tools, TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration

FILEPATH_INSTANCE = 'results/instances_'
FILEPATH_RESULT = 'results/result_'
FILEPATH_V = 'results/value_functions/'
MAX_TARGET_PROB = 0.9

# for i in range(108):
# args = {'instance': '01', 'method': 'vi', 'time': '0-00:05:00',
#         'job_id': 1, 'array_id': i, 'x': 0}
# args = tools.DotDict(args)


def main(raw_args=None):
    args = tools.load_args(raw_args)
    inst = pd.read_csv(FILEPATH_INSTANCE + args.instance + '.csv')
    cols = ['t', 'c', 'r', 'lab', 'mu']
    inst.loc[:, cols] = inst.loc[:, cols].applymap(tools.strip_split)

    if args.array_id - 1 + args.x >= len(inst):
        print('No more instances to solve within', args.time,
              'index:', args.array_id - 1 + args.x)
        exit(0)
    inst = inst.iloc[args.array_id - 1 + args.x]

    v_file = ('v_' + args.instance + '_' + str(inst[0]) + '_vi.npz')
    pi_file = ('pi_' + args.instance + '_' + str(inst[0]) + '_vi.npz')
    if ((v_file in os.listdir(FILEPATH_V)) and
            (pi_file not in os.listdir(FILEPATH_V))):
        env = Env(J=inst.J, S=inst.S, D=inst.D, gamma=inst.gamma,
                  e=inst.e, t=inst.t, c=inst.c, r=inst.r, P=inst.P,
                  lab=inst.lab, mu=inst.mu, max_time=args.time,
                  convergence_check=10)
        pi_learner = PolicyIteration()
        learner = ValueIteration(env, pi_learner)
        learner.V = np.load(FILEPATH_V + v_file)['arr_0']

        pi_learner = PolicyIteration()
        if args.method == 'vi':
            learner = ValueIteration(env, pi_learner)
            v_file = ('v_' + args.instance + '_' + str(inst[0]) + '_vi.npz')
            print('Loading V from file')
            learner.V = np.load(FILEPATH_V + v_file)['arr_0']
            pi_file = ('pi_' + args.instance + '_' + str(inst[0]) + '_vi.npz')
            if pi_file not in os.listdir(FILEPATH_V):
                learner.get_policy(env)
            if learner.Pi is not None:
                np.savez(FILEPATH_V + 'pi_' + args.instance + '_' +
                         str(inst[0]) + '_' + args.method + '.npz', learner.Pi)


if __name__ == '__main__':
    main()
