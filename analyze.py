"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import tools

INSTANCE_ID = '02'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCE_ID + '.csv'
FILEPATH_V = 'results/value_functions/'

inst = tools.inst_load(FILEPATH_INSTANCE)
tools.solved_and_left(inst)

methods = [column.split('_')[0] for column in inst.columns
           if column.endswith('job_id')]

opt_m = 'vi'
inst[opt_m + '_time'] = inst[opt_m + '_time'].map(
    lambda x: x if pd.isnull(x) else tools.get_time(x))

for method in methods:
    if method == opt_m:
        continue
    print(method)
    inst[method + '_time'] = inst[method + '_time'].map(
         lambda x: x if pd.isnull(x) else tools.get_time(x))
    inst_conv = inst[pd.notnull(inst[method + '_g']) &
                     pd.notnull(inst[opt_m + '_g'])]
    inst_part = inst[pd.notnull(inst[method + '_g']) |
                     pd.notnull(inst[opt_m + '_g'])]
    inst_tmp = inst[pd.notnull(inst[method + '_g_tmp']) &
                    pd.notnull(inst[opt_m + '_g_tmp'])]

    # plt.hist(inst_conv['ospi_opt_gap'])
    inst_conv.boxplot(column=method + '_opt_gap', by='gamma')
    plt.ylabel('Opt_gap')
    plt.title('Optimality Gap versus queues for ' + method + ' vs ' + opt_m)
    plt.show()
    # https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
    # plt.violinplot(inst_conv['ospi_opt_gap'], showmedians=True)

    plt.scatter(inst_conv['vi_time']/(60*60), inst_conv['ospi_opt_gap'])
    plt.scatter(inst_conv['ospi_time']/(60*60), inst_conv['ospi_opt_gap'])
    plt.xlabel('Running time (hours)')
    plt.ylabel('Optimality gap')
    plt.title('Running time vs. gap ' + method + ' vs ' + opt_m)
    plt.legend(['vi', 'ospi'])
    plt.show()

    inst_conv.boxplot(column=['vi_time', 'ospi_time'], by='load')
    plt.ylabel('Running time (sec.)')
    plt.suptitle('Running time (sec.) vs. Load')
    plt.show()

    plt.scatter(inst_conv['ospi_opt_gap']/(60*60), inst_conv['size'])
    plt.xlabel('Opt_gap')
    plt.ylabel('State space size')
    plt.title('Opt gap vs. State space size')
    plt.show()

    plt.scatter(inst_conv['vi_time']/(60*60), inst_conv['size'])
    plt.scatter(inst_conv['ospi_time']/(60*60), inst_conv['size'])
    plt.xlabel('Running time (hours)')
    plt.ylabel('State space size')
    plt.title('Running time vs. State space size')
    plt.legend(['vi', 'ospi'])
    plt.show()

    inst_conv.boxplot(column='ospi_opt_gap', by='load')
    plt.xlabel('load')
    plt.ylabel('Optimality Gap')
    plt.title('Load vs. Optimality Gap')
    plt.show()

    inst_conv[inst_conv['J'] == 2].boxplot(column='ospi_opt_gap', by='S')
    plt.title('Optimality Gap vs servers')
    plt.show()

    inst['solved'] = pd.notnull(inst['ospi_opt_gap'])
    inst.boxplot(column='size', by='solved')
    plt.ylabel('Size')
    plt.title('Size per solved')
    plt.show()

    inst_unsolved = inst[pd.isnull(inst['ospi_g']) | pd.isnull(inst['vi_g'])]
    unique_loads = inst['load'].unique()
    solved = inst_conv['load'].value_counts().reindex(unique_loads, fill_value=0)
    unsolved = inst_unsolved['load'].value_counts().reindex(unique_loads,
                                                            fill_value=0)
    print('unsolved: ', unsolved)
    print('solved:', solved)

    plt.bar(unique_loads, solved, width=0.05)
    plt.bar(unique_loads, unsolved, width=0.05, bottom=solved)
    plt.show()

# OPT_METHOD = 'vi'

for file in os.listdir(FILEPATH_V):
    if not (file.startswith('g_' + INSTANCE_ID) or file.endswith('_pi.npz')):
        continue
    print(file)
    # g_mem is just a list
    # pi_file = ('pi_' + args.instance + '_' + str(inst[0]) + '_pi.npz')
    # v_file = ('v_' + args.instance + '_' + str(inst[0]) + '_pi.npz')
    # g_file = ('g_' + args.instance + '_' + str(inst[0]) + '_pi.npz')
    # if ((pi_file in os.listdir(FILEPATH_V)) &
    #         (v_file in os.listdir(FILEPATH_V))):
    #     print('Loading pi & v from file', flush=True)
    #     Pi = np.load(FILEPATH_V + pi_file)['arr_0']
    #     V = np.load(FILEPATH_V + v_file)['arr_0']
    #     g_mem = np.load(FILEPATH_V + g_file)['arr_0']
    #     g_mem = learner.policy_iteration(env, g_mem=g_mem, Pi=Pi, V=V)
    # else:
    #     g_mem = learner.policy_iteration(env)
    # np.savez(FILEPATH_V + 'g_' + args.instance + '_' + str(inst[0]) + '_'
    #          + args.method + '.npz', g_mem)