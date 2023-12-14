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

start_K = 0  # int(1e3)
M = 50
alpha = 0.05

tools.remove_empty_files(FILEPATH_READ)

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
instance_names = [instance_names[9]]
# Debug
"""
instance_name = instance_names[0]
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
        N = len(kpi_df)
        if kpi_df['time'].iloc[0] < 1:
            kpi_df = kpi_df[start_K:].reset_index()
        kpi_df = kpi_df[kpi_df['time'] != 0]
        kpi_df = kpi_df.reindex(columns=[*kpi_df.columns.tolist(),
                                         'reward', 'g', 'target', 'avg_wait'])
        T = int((N - start_K) / M)
        inst.loc[row_id, ['N', 'start_K', 'batch_T']] = N, start_K, T
        t, r, c = (inst.loc[row_id, 't'], inst.loc[row_id, 'r'],
                   inst.loc[row_id, 'c'])
        for i in range(inst.loc[row_id, 'J']):
            mask = (kpi_df['class'] == i)
            kpi_df.loc[mask, 'reward'] = np.where(kpi_df.loc[mask, 'wait']
                                                  <= t[i], r[i], r[i] - c[i])
            kpi_df.loc[mask, 'target'] = \
                (np.where(kpi_df.loc[mask, 'wait'] <= t[i], 1, 0).cumsum()
                 / np.arange(1, sum(mask) + 1))
            kpi_df.loc[mask, 'avg_wait'] = (kpi_df.loc[mask, 'wait'].cumsum()
                                            / np.arange(1, sum(mask) + 1))
        # per admission
        kpi_df['perc'] = kpi_df['reward'].expanding().mean()
        MA = kpi_df['reward'].rolling(window=T).mean().values[T::T]
        ci_perc = tools.conf_int(alpha, MA)
        inst.loc[row_id, ['perc', 'ci_perc']] = kpi_df['perc'].iloc[-1], ci_perc
        # per time
        kpi_df['g'] = (kpi_df['reward'].cumsum() /
                       (kpi_df['time'] - kpi_df.loc[0, 'time']))
        times = kpi_df['time'].values[T::T] - kpi_df['time'].values[::T][:-1]
        MA = kpi_df['reward'].rolling(window=T).sum().values[T::T] / times
        ci_g = tools.conf_int(alpha, MA)
        inst.loc[row_id, ['g', 'ci_g']] = kpi_df['g'].iloc[-1], ci_g
        pkl.dump([arr_times, fil, heap, kpi_df, s, time],
                 open(FILEPATH_PICKLES + file, 'wb'))
    inst.to_csv(FILEPATH_INSTANCE + instance_name, index=False)
    print('saved:', instance_name)
