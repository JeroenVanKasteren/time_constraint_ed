"""
Process all unread simulation result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os
import re
from utils import tools

# Debug
# args = {'instance': 'J2'}
# args = tools.DotDict(args)
args = tools.load_args()

FILEPATH_INSTANCE = 'results/instances_' + args.instance + '_sim.csv'
FILEPATH_RESULT = 'results/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

inst_nr = ''  # '' to loop over all
method = 'not specified'
start_K = int(1e2)
M = 100
alpha = 0.05

if method == 'not specified':
    methods = ['ospi', 'cmu_t_min', 'cmu_t_max', 'fcfs', 'sdf',
               'sdfprior', 'l_max', 'l_min']
else:
    methods = [method]

pattern = 'result_' + args.instance + '_'
pattern += r'\d' if inst_nr == '' else ''
files = [file for file in os.listdir(FILEPATH_PICKLES)
         if (re.match('result_' + args.instance + r'_\d', file) is not None)]

inst = tools.inst_load(FILEPATH_INSTANCE)
for file in files:
    row_id = int(file.split('_')[2])
    method = '_'.join(file.split('_')[3:])[:-4]
    pickle = pkl.load(open(FILEPATH_PICKLES + file, 'rb'))
    kpi_df = pd.DataFrame(pickle['kpi'], columns=['time', 'class', 'wait'])
    kpi_df = kpi_df[kpi_df['time'] != 0]
    kpi_df = kpi_df.reindex(columns=[*kpi_df.columns.tolist(), 'reward'])
    N = len(kpi_df)
    T = (N - start_K) // M
    K = start_K + (N - start_K) % M
    inst.loc[row_id, ['N', 'start_K', 'batch_T']] = N, K, T
    t, r, c = (inst.loc[row_id, 't'], inst.loc[row_id, 'r'],
               inst.loc[row_id, 'c'])
    for i in range(inst.loc[row_id, 'J']):
        mask = (kpi_df['class'] == i)
        kpi_df.loc[mask, 'reward'] = np.where(kpi_df.loc[mask, 'wait']
                                              <= t[i], r[i], r[i] - c[i])
    pickle['kpi'] = kpi_df
    # per admission
    MA = tools.moving_average_admission(kpi_df, K, M)
    ci_perc = tools.conf_int(alpha, MA)
    inst.loc[row_id, [method+'_perc', method+'_perc_ci']] = MA.mean(), ci_perc
    # per time
    MA, _ = tools.moving_average(kpi_df, K, M, T)
    ci_g = tools.conf_int(alpha, MA)
    inst.loc[row_id, [method+'_g', method+'_g_ci']] = MA.mean(), ci_g
    pkl.dump(pickle, open(FILEPATH_PICKLES + file, 'wb'))

inst.to_csv(FILEPATH_INSTANCE, index=False)
print('saved:', FILEPATH_INSTANCE)

tools.solved_and_left(inst, sim=True)
