"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import os
import utils
from utils import TimeConstraintEDs as Env, PolicyIteration, \
    ValueIteration, OneStepPolicyImprovement, plot_pi, plot_v

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values
env = utils.env.TimeConstraintEDs

# --------------------- Plotting ---------------------
instances = [instance_names[i - 1] for i in range(1, 7)]
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'g', False)
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'perc')
instances = [instance_names[i - 1] for i in range(7, 12)]
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'g', False)
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'perc')
instances = [instance_names[i - 1] for i in range(12, 15)]
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'g', False)
utils.plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods,
                              'perc')

# method = 'fcfs'  # method = 1
# instance_name = instance_names[10]
# method, row_id, inst, pickle = utils.tools.load_result(method, instance_name)
#
# utils.plotting.plot_convergence(pickle['kpi'], method,
#                                 inst.loc[row_id, 'start_K'],
#                                 inst.loc[row_id, 'batch_T'],
#                                 m=100)

# start = 0
# start = round_significance(random.randint(0, len(kpi_df)-size), 2)
# utils.plotting.plot_waiting(inst.loc[row_id], pickle['kpi'], 1000, start)

# Debugging by comparing theoretical results
# create instance in instance_sim_gen with J=1 and S=5

# Use instance 3 with mu_j = mu for all j and compare with FCFS


def theory(inst_row, gamma):
    """
    Calculates statistics of
    (Checked with Erlang C calculator, M/M/s)

    parameters
        inst_row: one row of instance dataframe
        gamma: time scaling factor

    return
        g         (float): long term average reward
        exp_wait  (float): expected waiting time, E(W_q)
        tail_prob (float): P(W>t)
    """
    g = 0
    lab = sum(inst_row.lab)
    pi_0 = env.get_pi_0(gamma, inst_row.S, inst_row.load, lab)
    block_prob = pi_0 / (1 - inst_row.load)
    exp_wait = block_prob / (inst_row.S * inst_row.mu[0] - lab)
    success_prob = []
    for i in range(inst_row.J):
        prob_i = inst_row.lab[i] / lab
        tail_prob_i = env.get_tail_prob(gamma, inst_row.S, inst_row.load, lab,
                                        inst_row.mu[i], pi_0,
                                        inst_row.t[i]*gamma)
        # identical
        # tail_prob_i = block_prob * np.exp(-(inst_row.S * inst_row.mu[i] - lab)
        #                                   * inst_row.t[i])
        g += prob_i * lab * (inst_row.r[i] - inst_row.c[i] * tail_prob_i)
        success_prob.append(1 - tail_prob_i)
    return exp_wait, g, success_prob


# interested = [instance_names[i - 1] for i in [1, 3, 9, 10, 11, 12]]
# interested = [instance_names[i - 1] for i in [9]]
for instance_name in instance_names:
    inst = utils.tools.inst_load(FILEPATH_INSTANCE + instance_name)
    exp_wait, g, success_prob = theory(inst.loc[0], 1e6)
    print(f'inst: {instance_name} \n'
          f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
          f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
          f'P(W<t) = {["%.4f" % elem for elem in success_prob]}')
    for i in range(len(methods)):
        method, row_id, inst, pickle = (
            utils.tools.load_result(i, instance_name))
        kpi_df, time, arr, dep = (pickle['kpi'], pickle['time'],
                                       pickle['arr'], pickle['dep'])
        reward_per_class = kpi_df.groupby('class')['reward'].mean()
        print(instance_name, method)
        print(f'Arrival rates: {sum(inst.lab[0]):0.4f} <> '
              f'{len(kpi_df)/(kpi_df.time.values[-1]):0.4f} after '
              f'{len(kpi_df)} sims in {time/60:0.2f} hours\n'
              f'Sim g: {inst.loc[i].g:0.4f} +/- {inst.loc[i,"ci_g"]:0.4f}'
              f' weighted average: '
              f'{sum(reward_per_class * inst.lab[i]):0.4f} \n'
              f'Sim E(W)={kpi_df["wait"].mean():0.4f}'
              f' Sim P(W < t)={inst.loc[i].perc:0.4f}'
              f' +/- {inst.loc[i,"ci_perc"]:0.4f}')
        print(f'{reward_per_class}')
        print('-'*10, '\n')
    print('-'*120, '\n', '-'*120, '\n')



