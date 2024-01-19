"""
Process all unread simulation result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os
from utils import tools

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

start_K = int(1e2)
M = 100
alpha = 0.05

tools.remove_empty_files(FILEPATH_READ)

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
# instance_names = instance_names[10:12]
# instance_names = [instance_names[9]]

# Debug
"""
instance_name = instance_names[8]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
instance_id = instance_name.split('_')[2][:-4]
methods = inst['method'].values
row_id = 0
method = methods[row_id]
"""

for instance_name in instance_names:
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
    instance_id = instance_name.split('_')[2][:-4]
    methods = inst['method'].values
    for row_id, method in enumerate(methods):
        file = 'result_' + instance_id + '_' + method + '.pkl'
        arr_times, fil, heap, kpi, s, time = \
            pkl.load(open(FILEPATH_PICKLES + file, 'rb'))
        kpi_df = pd.DataFrame(kpi, columns=['time', 'class', 'wait'])
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
        # per admission
        MA = kpi_df['reward'].values[K:].reshape(-1, M).mean(axis=0)
        ci_perc = tools.conf_int(alpha, MA)
        inst.loc[row_id, ['perc', 'ci_perc']] = MA.mean(), ci_perc
        # per time
        times = kpi_df['time'].values[K+T-1::T] - kpi_df['time'].values[K::T]
        MA = kpi_df['reward'].values[K:].reshape(-1, M).sum(axis=0) / times
        ci_g = tools.conf_int(alpha, MA)
        inst.loc[row_id, ['g', 'ci_g']] = MA.mean(), ci_g
        pkl.dump([arr_times, fil, heap, kpi_df, s, time],
                 open(FILEPATH_PICKLES + file, 'wb'))
    inst.to_csv(FILEPATH_INSTANCE + instance_name, index=False)
    print('saved:', instance_name)
