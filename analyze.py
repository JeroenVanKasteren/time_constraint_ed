"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from utils import plotting, tools, TimeConstraintEDs as Env

INSTANCES_ID = 'J1'
FILEPATH_INSTANCE = 'results/instances_' + INSTANCES_ID
FILEPATH_V = 'results/value_functions/'

inst_set = tools.inst_load(FILEPATH_INSTANCE + '.csv')
inst_set['smu(1-rho)'] = tools.get_smu_rho(inst_set.lab,
                                           inst_set.mu,
                                           inst_set.S)
tools.solved_and_left(inst_set)
methods = ['_'.join(column.split('_')[:-2]) for column in inst_set.columns
           if column.endswith('job_id')]
if 'instances_' + INSTANCES_ID + '_sim.csv' in os.listdir('results'):
    sim_set = tools.inst_load(FILEPATH_INSTANCE + '_sim.csv')
    tools.solved_and_left(sim_set, sim=True)
    sim_methods = ['_'.join(column.split('_')[:-1]) for column in
                   sim_set.columns if column.endswith('_g')]
    sim_set['smu(1-rho)'] = tools.get_smu_rho(sim_set.lab,
                                              sim_set.mu,
                                              sim_set.S)

# inst_set[method + '_time'] = inst_set[method + '_time'].map(
    #     lambda x: x if pd.isnull(x) else tools.get_time(x))
# inst_conv = inst_set[pd.notnull(inst_set[method + '_g']) &
#                      pd.notnull(inst_set[opt_m + '_g'])]
# inst_part = inst_set[pd.notnull(inst_set[method + '_g']) |
#                      pd.notnull(inst_set[opt_m + '_g'])]
# inst_tmp = inst_set[pd.notnull(inst_set[method + '_g_tmp']) &
#                     pd.notnull(inst_set[opt_m + '_g_tmp'])]

opt_m = 'vi'
opt_gap = {}
for method in [m for m in methods if m != opt_m]:
    subset = inst_set[[method + '_g', opt_m + '_g']].dropna()
    inst_conv = subset[pd.notnull(subset[method + '_g']) &
                       pd.notnull(subset[opt_m + '_g'])]
    opt_gap[method] = (abs(subset[method + '_g'] - subset[opt_m + '_g'])
                       / subset[opt_m + '_g'])

ref_m = 'ospi'
rel_opt_gap = {}
for i, method in enumerate([m for m in methods if m != ref_m]):
    cols = list({method + '_g', ref_m + '_g', opt_m + '_g'})
    subset = inst_set[cols].dropna()
    rel_opt_gap[method] = ((subset[method + '_g'] - subset[ref_m + '_g'])
                           / subset[opt_m + '_g'])

fig, ax = plt.subplots()
for i, method in enumerate([m for m in methods if m != opt_m]):
    ax.boxplot(opt_gap[method], positions=[i], tick_labels=[method])
ax.set_title('Optimality gap for ' + INSTANCES_ID)
plt.show()

fig, ax = plt.subplots()
x_lab, y_lab = 'size', 'smu(1-rho)'
for i, method in enumerate([m for m in methods if m != ref_m]):
    cols = list({x_lab, y_lab, method + '_g', ref_m + '_g', opt_m + '_g'})
    subset = inst_set[cols].dropna()
    plotting.plot_xyc(subset[x_lab], subset[y_lab], rel_opt_gap[method],
                      title=(ref_m + ' vs ' + method + ' rel to opt for '
                             + INSTANCES_ID),
                      x_lab=x_lab,
                      y_lab=y_lab,
                      c_lab='rel opt gap')

for i, method in enumerate([m for m in methods if m != opt_m]):
    ax.boxplot(rel_opt_gap[method], positions=[i], tick_labels=[method])
    plt.axhline(0)
    plt.title(ref_m + ' vs methods rel to opt for ' + INSTANCES_ID)
    plt.show()


# plt.hist(inst_conv['ospi_opt_gap'])
# inst_conv.boxplot(column=method + '_opt_gap', by='gamma')
# plt.ylabel('Opt_gap')
# plt.title('Optimality Gap versus queues for ' + method + ' vs ' + opt_m)
# plt.show()
# https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html
# plt.violinplot(inst_conv['ospi_opt_gap'], showmedians=True)

# plt.scatter(inst_conv[opt_m+'_time']/(60*60), inst_conv[method+'_opt_gap'])
# plt.scatter(inst_conv[opt_m+'_time']/(60*60), inst_conv[method+'_opt_gap'])
# plt.xlabel('Running time (hours)')
# plt.ylabel('Optimality gap')
# plt.title('Running time vs. gap for ' + method + ' vs ' + opt_m)
# plt.legend(['vi', 'ospi'])
# plt.show()

# inst_conv.boxplot(column=[opt_m+'_time', method+'_time'], by='load')
# plt.ylabel('Running time (sec.)')
# plt.suptitle('Running time (sec.) vs. Load for ' + method + ' vs ' + opt_m)
# plt.show()

# plt.scatter(inst_conv[method+'_opt_gap']/(60*60), inst_conv['size'])
# plt.xlabel('Opt_gap')
# plt.ylabel('State space size')
# plt.title('Opt gap vs. State space size for '+method)
# plt.show()

# plt.scatter(inst_conv[opt_m+'_time']/(60*60), inst_conv['size'])
# plt.scatter(inst_conv[method+'_time']/(60*60), inst_conv['size'])
# plt.xlabel('Running time (hours)')
# plt.ylabel('State space size')
# plt.title('Running time vs. State space size')
# plt.legend([opt_m, method])
# plt.show()

# inst_conv.boxplot(column=method+'_opt_gap', by='load')
# plt.xlabel('load')
# plt.ylabel('Optimality Gap')
# plt.title('Load vs. Optimality Gap for ' + method)
# plt.show()
#
# inst_conv[inst_conv['J'] == 2].boxplot(column=method+'_opt_gap', by='S')
# plt.title('Optimality Gap vs servers for ' + method)
# plt.show()

# inst['solved'] = pd.notnull(inst[method+'_opt_gap'])
# inst.boxplot(column='size', by='solved')
# plt.ylabel('Size')
# plt.title('Size per solved')
# plt.show()

inst_unsolved = inst[pd.isnull(inst[method+'_g']) | pd.isnull(inst[opt_m+'_g'])]
unique_loads = inst['load'].unique()
solved = inst_conv['load'].value_counts().reindex(unique_loads, fill_value=0)
unsolved = inst_unsolved['load'].value_counts().reindex(unique_loads,
                                                        fill_value=0)
print('unsolved: ', unsolved)
print('solved:', solved)

plt.bar(unique_loads, solved, width=0.05)
plt.bar(unique_loads, unsolved, width=0.05, bottom=solved)
plt.show()


for file in os.listdir(FILEPATH_V):
    if not (file.startswith('g_' + INSTANCES_ID)):
        continue  # or file.endswith('_pi.npz')):
    print(file)
    file = 'g_02_13_pi.npz'
    g_mem = np.load(FILEPATH_V + file)['arr_0']

cap_d = 100
solve_id = 0
inst = tools.inst_load(FILEPATH_INSTANCE + '.csv')
inst = inst.iloc[0]
env = Env(J=inst.J, S=inst.S, D=inst.D,
          gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
          mu=inst.mu, lab=inst.lab)
for method in methods:
    pi_file = ('pi_' + INSTANCES_ID + '_' + str(solve_id) + '_' +
               method + '.npz')
    Pi = np.load(FILEPATH_V + pi_file)['arr_0']
    name = method + '_' + str(solve_id)
    state = np.concatenate(([0],
                            [0] * inst.J,
                            [0, int(inst.S / 2)])).astype(object)
    plotting.plot_pi(env, Pi, False, state=state, name=name, t=inst.t)

    v_file = ('v_' + INSTANCES_ID + '_' + str(solve_id) + '_' +
              method + '.npz')
    v = np.load(FILEPATH_V + v_file)['arr_0']
    state = np.concatenate(([0] * inst.J,
                            [0, int(inst.S / 2)])).astype(object)
    name = method + '_' + str(solve_id)
    plotting.plot_v(env, v, False, state=state, name=name)
