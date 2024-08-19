"""
Load and visualize results of simulation.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import numpy as np
import os
from utils import plotting, tools, TimeConstraintEDs as Env, PolicyIteration

FILEPATH_INSTANCE = 'results/'
FILEPATH_READ = 'results/read/'
FILEPATH_PICKLES = 'results/simulation_pickles/'
FILEPATH_V = 'results/value_functions/'
INSTANCES_ID = 'J1'
FILEPATH_INSTANCES = 'results/instances_' + INSTANCES_ID + '.csv'

instance_names = [f for f in os.listdir(FILEPATH_INSTANCE)
                  if f.startswith('instance_sim_')]
inst = tools.inst_load(FILEPATH_INSTANCE + instance_names[0])
methods = inst['method'].values

# --------------------- Plotting ---------------------
instances = [instance_names[i - 1] for i in range(1, 7)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')
instances = [instance_names[i - 1] for i in range(7, 12)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')
instances = [instance_names[i - 1] for i in range(12, 15)]
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'g', False)
plotting.plot_multi_bar(FILEPATH_INSTANCE, instances, methods, 'perc')

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



# interested = [instance_names[i - 1] for i in [1, 3, 9, 10, 11, 12]]
# interested = [instance_names[i - 1] for i in [9]]
for instance_name in instance_names:
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_name)
    exp_wait, g, success_prob = theory(inst.loc[0], 1e6)
    print(f'inst: {instance_name} \n'
          f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
          f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
          f'P(W<t) = {["%.4f" % elem for elem in success_prob]}')
    # for i in range(len(methods)):
    #     method, row_id, inst, pickle = (
    #         tools.load_result(i, instance_name))
    #     kpi_df, time, arr, dep = (pickle['kpi'], pickle['time'],
    #                                    pickle['arr'], pickle['dep'])
    #     reward_per_class = kpi_df.groupby('class')['reward'].mean()
    #     print(instance_name, method)
    #     print(f'Arrival rates: {sum(inst.lab[0]):0.4f} <> '
    #           f'{len(kpi_df)/(kpi_df.time.values[-1]):0.4f} after '
    #           f'{len(kpi_df)} sims in {time/60:0.2f} hours\n'
    #           f'Sim g: {inst.loc[i].g:0.4f} +/- {inst.loc[i,"ci_g"]:0.4f}'
    #           f' weighted average: '
    #           f'{sum(reward_per_class * inst.lab[i]):0.4f} \n'
    #           f'Sim E(W)={kpi_df["wait"].mean():0.4f}'
    #           f' Sim P(W < t)={inst.loc[i].perc:0.4f}'
    #           f' +/- {inst.loc[i,"ci_perc"]:0.4f}')
    #     print(f'{reward_per_class}')
    #     print('-'*10, '\n')
    # print('-'*120, '\n', '-'*120, '\n')

cap_d = 100
solve_id, inst_id = 57, 13-1
for solve_id, inst_id in zip([8, 57, 93], [12-1, 13-1, 14-1]):
    print(f'solve_id: {solve_id}, inst_id: {inst_id}')
    inst = tools.inst_load(FILEPATH_INSTANCE + instance_names[inst_id])
    inst = inst.iloc[0]
    env = Env(J=inst.J, S=inst.S, D=inst.D,
              gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
              mu=inst.mu, lab=inst.lab)
    for method in ['vi', 'ospi']:
        pi_file = ('pi_' + INSTANCES_ID + '_' + str(solve_id) + '_' +
                   method + '.npz')
        Pi = np.load(FILEPATH_V + pi_file)['arr_0']

        pi_learner = PolicyIteration(Pi=Pi)
        tools.summarize_policy(env, pi_learner, print_per_time=False)
        state = np.concatenate(([0],
                                [0]*inst.J,
                                [0, int(inst.S/2)])).astype(object)
        name = method + '_' + str(solve_id) + '_' + str(inst_id + 1)
        plotting.plot_pi(env, Pi, False, state=state, name=name)

        v_file = ('v_' + INSTANCES_ID + '_' + str(solve_id) + '_' +
                  method + '.npz')
        v = np.load(FILEPATH_V + v_file)['arr_0']
        state = np.concatenate(([0] * inst.J,
                                [0, int(inst.S / 2)])).astype(object)
        name = method + '_' + str(solve_id) + '_' + str(inst_id + 1)
        plotting.plot_v(env, v, False, state=state, name=name)
