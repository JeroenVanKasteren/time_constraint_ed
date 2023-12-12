"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
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
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_name)
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


start = 0  # round_significance(random.randint(0, len(kpi_df)-size), 2)
utils.plotting.plot_waiting(inst.loc[row_id], kpi_df, 1000, start)

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

# Debugging by comparing theoretical results
# create instance in instance_sim_gen with J=1 and S=5
# Use instance 3 with mu_j = mu for all j and compare with FCFS
inst_row = inst.loc[row_id]
for i in range(inst_row.J):
    lab_i = inst_row.lab[i] / sum(inst_row.lab)
    pi_0 = utils.env.get_pi_0(inst_row.gamma, inst_row.S, inst_row.load, lab_i)
    utils.env.get_tail_prob(inst_row.gamma, inst_row.S, inst_row.load, lab_i,
                            pi_0, inst_row.t[i])
