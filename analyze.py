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
FILEPATH_RESULTS = 'results/'
FILEPATH_V = 'results/value_functions/'
multi_xyc = True
violin = True

# Load in instance_set and extract methods used
instance_name = 'instances_' + INSTANCES_ID
inst_set = tools.inst_load(FILEPATH_RESULTS + instance_name + '.csv')
is_sim = 'sim' in INSTANCES_ID
tools.solved_and_left(inst_set, sim=is_sim)
solve_methods = []
if not is_sim:
    solve_methods = ['_'.join(column.split('_')[:-2])
                     for column in inst_set.columns
                     if column.endswith('job_id')]

# If a solve set, load in corresponding simulation set and append to inst_set
has_sim = False
sim_methods = []
if ((not is_sim) and
        (instance_name + '_sim.csv' in os.listdir(FILEPATH_RESULTS))):
    has_sim = True
    sim_set = tools.inst_load(FILEPATH_RESULTS + instance_name + '_sim.csv')
    tools.solved_and_left(sim_set, sim=True)
    sim_methods = ['_'.join(column.split('_')[:-1])
                   for column in sim_set.columns
                   if column.endswith('_g')]
    suffixes = ['_g', '_g_ci', '_perc', '_perc_ci']
    for method in sim_methods:
        inst_set[[method + x + '_sim' for x in suffixes]] = (
            sim_set[[method + x for x in suffixes]].copy())
methods_both = list(set(solve_methods) & set(sim_methods))

inst_set['smu(1-rho)'] = tools.get_smu_rho(inst_set.lab,
                                           inst_set.mu,
                                           inst_set.S)

# Difference between theory vs solve and sim for mu == mu_j
inst_theory = inst_set[inst_set.apply(lambda x_row: len(set(x_row.mu)) == 1,
                                      axis=1)].reindex()
theory_g = []
theory_g_gam = []
for i, row in inst_theory.iterrows():
    prob_i = row.lab/sum(row.lab)
    _, _, _, prob_late = tools.get_erlang_c(row.mu[0], row.load, row.S,
                                            row.t, prob_i)
    theory_g.append(sum(row.lab * (row.r - row.c * prob_late)))
    _, g, _ = tools.get_erlang_c_gam(row, row.gamma)
    theory_g_gam.append(g)
    # optional test:
    # _, g, _ = tools.get_erlang_c_gam(row, 1e6)
    # theory_g2.append(g)
inst_theory.loc[:, 'theory_g'] = theory_g
inst_theory.loc[:, 'theory_g_gam'] = theory_g_gam

# theory vs sim
plotting.plot_gap(inst_theory,
                  sim_methods,
                  '_g_sim',
                  'theory',
                  '_g',
                  'theory',
                  '_g',
                  'Gap of g for sims vs theory',
                  multi_xyc=multi_xyc,
                  violin=violin)
# theory discr. vs solve
plotting.plot_gap(inst_theory,
                  solve_methods,
                  '_g',
                  'theory',
                  '_g_gam',
                  'theory',
                  '_g_gam',
                  'Gap of g for solve vs discr theory',
                  multi_xyc=multi_xyc,
                  violin=violin)
# theory vs solve
plotting.plot_gap(inst_theory,
                  solve_methods,
                  '_g',
                  'theory',
                  '_g',
                  'theory',
                  '_g',
                  'Gap of g for solve vs theory',
                  multi_xyc=multi_xyc,
                  violin=violin)

# compare VI vs solve
plotting.plot_gap(inst_set,
                  solve_methods,
                  '_g',
                  'vi',
                  '_g',
                  'vi',
                  '_g',
                  'Gap of g for solve vs VI',
                  multi_xyc=multi_xyc,
                  violin=violin)

# compare VI vs sim
plotting.plot_gap(inst_set,
                  sim_methods,
                  '_g_sim',
                  'vi',
                  '_g',
                  'vi',
                  '_g',
                  'Gap of g for sim vs VI',
                  multi_xyc=multi_xyc,
                  violin=violin)

# compare OSPI vs solve (rel to vi)
plotting.plot_gap(inst_set,
                  solve_methods,
                  '_g',
                  'ospi',
                  '_g',
                  'vi',
                  '_g',
                  'Gap of g for sim vs OSPI rel to VI',
                  multi_xyc=multi_xyc,
                  violin=violin)

# exp_wait, g, success_prob = tools.get_gen_erlang_c(inst_row, 1e6)
# print(f'inst: {instance_name} \n'
#       f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
#       f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
#       f'P(W<t) = {["%.4f" % elem for elem in success_prob]}')

# to doorspitten
chosen_files = ['J1_2', 'J1_5', 'J2_8']  # ID_i
for file_id in chosen_files:
    file = 'g_' + file_id + '_pi.npz'
    g_mem = np.load(FILEPATH_V + 'g_' + file)['arr_0']

# find file
# for file in os.listdir(FILEPATH_V):
#     if not (file.startswith('g_' + INSTANCES_ID)):
#         continue  # or file.endswith('_pi.npz')):
#     print(file)


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
