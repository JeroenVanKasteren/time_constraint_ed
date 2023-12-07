"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import random
import utils

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
# instance_names = instance_names[1:6]

inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values

utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods, 'g')
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods,
                              'perc')


instance_name = instance_names[4]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
instance_id = instance_name.split('_')[2][:-4]
methods = inst['method'].values
row_id = 1
method = methods[row_id]
file = 'result_' + instance_id + '_' + method + '.pkl'
arr_times, fil, heap, kpi_df, s, time = (
    pkl.load(open(FILEPATH_PICKLES + file, 'rb')))

plt.scatter(kpi_df['time']/60, kpi_df['g'])
plt.xlabel('Running time (hours)')
plt.ylabel('g')
plt.title('g vs. time')
plt.show()

size = 200
start = 0  # round_significance(random.randint(0, len(kpi_df)-size), 2)
kpi_df_tmp = kpi_df[start:start+size]
x = np.arange(inst.J[0])
ys = [i + x + (i * x) ** 2 for i in range(inst.J[0])]
colors = plt.cm.rainbow(np.linspace(0, 1, len(ys)))
for i in range(inst.loc[row_id, 'J']):
    mask = (kpi_df_tmp['class'] == i)
    plt.scatter(kpi_df_tmp.loc[mask, 'time']/60, kpi_df_tmp.loc[mask, 'wait'],
                marker='x', label=i, color=colors[i])
    plt.axhline(y=inst.t[0][i], color=colors[i], linestyle='-')

plt.xlabel('Time (hours)')
plt.ylabel('wait')
plt.title('Waiting time per class')
plt.legend(loc='upper left')
plt.show()


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

