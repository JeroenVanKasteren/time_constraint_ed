"""
Process all unread simulation result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy.stats import norm
from utils import tools

INSTANCE_ID = '01'
FILEPATH_INSTANCE = 'results/instance_sim_' + INSTANCE_ID + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

start_K = int(1e4)
M = 50
alpha = 0.05

inst = tools.inst_load(FILEPATH_INSTANCE)
methods = inst['method'].values

tools.remove_empty_files(FILEPATH_READ)

# Process all unread result files
# file = 'result_' + INSTANCE_ID + '_sdf.pkl'
for file in os.listdir(FILEPATH_PICKLES):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    method = file.split('_')[2][:-4]
    row_id = (inst['method'] == method).idxmax()
    _, _, _, kpi, _, _ = pkl.load(open(FILEPATH_PICKLES + file, 'rb'))
    kpi_df = pd.DataFrame(kpi, columns=['time', 'class', 'wait'])
    N = len(kpi_df)
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
        kpi_df.loc[mask, 'reward'] = np.where(kpi_df.loc[mask, 'wait'] <= t[i],
                                              r[i], r[i] - c[i])
        kpi_df.loc[mask, 'target'] = \
            (np.where(kpi_df.loc[mask, 'wait'] <= t[i], 1, 0).cumsum()
             / np.arange(1, sum(mask) + 1))
        kpi_df.loc[mask, 'avg_wait'] = (kpi_df.loc[mask, 'wait'].cumsum()
                                        / np.arange(1, sum(mask) + 1))
    kpi_df['g'] = (kpi_df['reward'].cumsum() /
                   (kpi_df['time'] - kpi_df.loc[0, 'time']))
    # per admission
    # MA = kpi_df['wait'].rolling(window=T).mean().values[T::T]
    # per time
    times = kpi_df['time'].values[T::T] - kpi_df['time'].values[::T][:-1]
    MA = kpi_df['reward'].rolling(window=T).sum().values[T::T] / times
    conf_int = norm.ppf(1-alpha/2) * MA.std() / np.sqrt(len(MA))
    # MA.mean()
    inst.loc[row_id, ['g', 'conf_int']] = kpi_df['g'].iloc[-1], conf_int

# K analyses
# import matplotlib.pyplot as plt
# MA = kpi_df['wait'].rolling(window=T).mean().iloc[T::T]
# plt.scatter(np.arange(len(MA)), MA)
# plt.show()
#
# plt.scatter(np.arange(len(MA)), MA.cumsum()/np.arange(len(MA)))
# plt.show()
#
# plt.scatter(np.arange(len(kpi_df)), kpi_df['g'])
# plt.show()

inst.to_csv(FILEPATH_INSTANCE, index=False)
