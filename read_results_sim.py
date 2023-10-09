"""
Process all unread simulation result files.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import pandas as pd
import pickle as pkl
import os
from utils import tools

INSTANCE_ID = '01'
FILEPATH_INSTANCE = 'results/instance_sim_' + INSTANCE_ID + '.csv'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

start_K = 1e3
batch_T = 1e4

inst = pd.read_csv(FILEPATH_INSTANCE)
methods = inst['method'].values

tools.remove_empty_files(FILEPATH_READ)

# Process all unread result files
for file in os.listdir(FILEPATH_PICKLES):
    if not file.startswith('result_' + INSTANCE_ID):
        continue
    method = file.split('_')[2][:-4]
    _, _, _, kpi, _, _ = pkl.load(open(pickle_file, 'rb'))
    kpi_df = pd.DataFrame(kpi, columns=['time', 'class', 'wait'])
    kpi_df = kpi_df[kpi_df['time'] != 0]
    kpi_df = kpi_df.reindex(columns=[*kpi_df.columns.tolist(),
                                     'reward', 'g', 'target', 'avg_wait'])
    N = len(kpi_df)
    for i in range(J):
        mask = (kpi_df['class'] == i)
        kpi_df.loc[mask, 'reward'] = np.where(kpi_df.loc[mask, 'wait'] <= t[i],
                                              r[i], r[i] - c[i])
        kpi_df.loc[mask, 'target'] = \
            (np.where(kpi_df.loc[mask, 'wait'] <= t[i], 1, 0).cumsum()
             / np.arange(1, sum(mask) + 1))
        kpi_df.loc[mask, 'avg_wait'] = (kpi_df.loc[mask, 'wait'].cumsum()
                                        / np.arange(1, sum(mask) + 1))
    kpi_df['g'] = (kpi_df['reward'].cumsum() / kpi_df['time'])