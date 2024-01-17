"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import os
import utils

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values


utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods,
                              'g', True)
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods,
                              'perc')

method = 'ospi'  # method = 1
instance_name = instance_names[0]
method, row_id, inst, (arr_times, fil, heap, kpi_df, s, time) = (
    utils.tools.load_result(method, instance_name))

utils.plotting.plot_convergence(kpi_df, method)

start = 0
# start = round_significance(random.randint(0, len(kpi_df)-size), 2)
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

# def get_g(inst_row, i):
#     lab = sum(inst_row.lab) if i == None else inst_row.lab[i]
#     pi_0 = env.get_pi_0(inst_row.gamma, inst_row.S, inst_row.load, lab)
#     tail_prob = env.get_tail_prob(inst_row.gamma, inst_row.S, inst_row.load,
#                                   lab, inst_row.mu[i], pi_0,
#                                   inst_row.t[i])
#     g = (inst_row.r[i] - inst_row.c[i] * tail_prob) * inst_row.lab[i]

# Use instance 3 with mu_j = mu for all j and compare with FCFS
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[2])
row_id = np.where(inst.method == 'fcfs')[0][0]
inst_row = inst.loc[row_id]
env = utils.env.TimeConstraintEDs


def get_g(inst_row, gamma):
    g = 0
    lab = sum(inst_row.lab)
    for i in range(inst_row.J):
        prob_i = inst_row.lab[i] / lab
        pi_0 = env.get_pi_0(gamma, inst_row.S, inst_row.load, lab)
        tail_prob = env.get_tail_prob(gamma, inst_row.S, inst_row.load, lab,
                                      inst_row.mu[i], pi_0, inst_row.t[i])
        g += prob_i * inst_row.lab[i] * (inst_row.r[i] -
                                         inst_row.c[i] * tail_prob)
    return g


for instance_name in instance_names:
    print(instance_name)
    inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_name)
    print(f'g bounds: [{sum(inst.r[0]*inst.lab[0])}, 0]')
    print(f'Heuristic lower bound: [{get_g(inst.loc[0], inst_row.gamma)}, 0]')
    print(f'Heuristic lower bound: [{get_g(inst.loc[0], 1e6)}, 0]')
    instance_id = instance_name.split('_')[2][:-4]
