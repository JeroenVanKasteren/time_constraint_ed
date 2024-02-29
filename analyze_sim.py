"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import os
import utils

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values
env = utils.env.TimeConstraintEDs

utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods,
                              'g', True)
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instance_names, methods,
                              'perc')

method = 'ospi'  # method = 1
instance_name = instance_names[0]
method, row_id, inst, pickle = utils.tools.load_result(method, instance_name)

utils.plotting.plot_convergence(pickle['kpi'], method,
                                inst.loc[row_id, 'start_K'],
                                inst.loc[row_id, 'batch_T'],
                                M = 100)

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


def theory(inst_row, gamma):
    g = 0
    lab = sum(inst_row.lab)
    pi_0 = env.get_pi_0(gamma, inst_row.S, inst_row.load, lab)
    block_prob = pi_0 / (1 - inst_row.load)
    exp_wait = block_prob / (inst_row.S * inst_row.mu[0] - lab)
    tail_prob = []
    for i in range(inst_row.J):
        prob_i = inst_row.lab[i] / lab
        tail_prob_i = env.get_tail_prob(gamma, inst_row.S, inst_row.load, lab,
                                        inst_row.mu[i], pi_0,
                                        inst_row.t[i]*gamma)
        # identical
        # tail_prob_i = block_prob * np.exp(-(inst_row.S * inst_row.mu[i] - lab)
        #                                   * inst_row.t[i])
        g += prob_i * inst_row.lab[i] * (inst_row.r[i] -
                                         inst_row.c[i] * tail_prob_i)
        tail_prob.append(1 - tail_prob_i)
    return g, exp_wait, tail_prob


# interested = [instance_names[i - 1] for i in [3, 9, 10, 11, 12]]
interested = [instance_names[i - 1] for i in [3]]
for instance_name in interested:
    inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_name)
    g, exp_wait, tail_prob = theory(inst.loc[0], 1e6)
    print(f'inst: {instance_name} \n'
          f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
          f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
          f'P(W<t) = {["%.4f" % elem for elem in tail_prob]}')
    for i in range(len(methods)):
        method, row_id, inst, (arr_times, fil, heap, kpi_df, s, time) = (
            utils.tools.load_result(i, instance_name))
        reward_per_class = kpi_df.groupby('class')['reward'].mean()
        print(instance_name, method)
        print(f'Arrival rates: {sum(inst.lab[0]):0.4f} <> '
              f'{len(kpi_df)/(kpi_df.time.values[-1]):0.4f} in {time:0.2f} '
              f'({kpi_df.time.values[-1]:0.2f})\n'
              f'Sim g: {inst.loc[i].g:0.4f} +/- {inst.loc[i,"ci_g"]:0.4f}'
              f' weighted average: '
              f'{sum(reward_per_class * inst.lab[i]):0.4f} \n'
              f'Sim E(W)={kpi_df["wait"].mean():0.4f}'
              f' Sim P(W < t)={inst.loc[i].perc:0.4f}'
              f' +/- {inst.loc[i,"ci_perc"]:0.4f}')
        print(f'{reward_per_class}')
    print('')
