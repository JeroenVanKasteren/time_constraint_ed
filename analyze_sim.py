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


interested = [instance_names[i - 1] for i in [3, 9, 10, 11, 12]]
env = utils.env.TimeConstraintEDs
for instance_name in interested:
    print(f'inst: {instance_name} \n'
          f'g upper bound: {sum(inst.r[i] * inst.lab[i]):0.4f} \n'
          f'Theory, gamma=1, g={get_g(inst.loc[i], inst.loc[i].gamma):0.4f}'
          f' gamma>>1, {get_g(inst.loc[i], 1e6):0.4f}')
    for i in range(len(methods)):
        method, row_id, inst, (arr_times, fil, heap, kpi_df, s, time) = (
            utils.tools.load_result(i, instance_name))
        # of kpi_df dataframe average of reward column per class
        #conditional reward
        reward_per_class = kpi_df.groupby('class')['reward'].mean()
        print(instance_name, method)
        print(f'result: {inst.loc[i].g:0.4f}')
        print(f'{reward_per_class} \n'
              f'weighted average: '
              f'{sum(reward_per_class * inst.lab[i]) / sum(inst.lab[i]):0.4f}'
              f' mean: {kpi_df["reward"].mean():0.4f}')
    print('')

print(sum(kpi_df['reward']) / (kpi_df['time'].iloc[-1] - kpi_df.loc[0, 'time']))
# kpi_df['g'] = (kpi_df['reward'].cumsum() /
#                (kpi_df['time'] - kpi_df.loc[0, 'time']))
# times = kpi_df['time'].values[T::T] - kpi_df['time'].values[::T][:-1]
# MA = kpi_df['reward'].rolling(window=T).sum().values[T::T] / times
# ci_g = tools.conf_int(alpha, MA)
# inst.loc[row_id, ['g', 'ci_g']] = kpi_df['g'].iloc[-1], ci_g
