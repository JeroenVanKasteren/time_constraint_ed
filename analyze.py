"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from utils import plotting, tools, TimeConstraintEDs as Env, PolicyIteration

INSTANCES_ID = 'J1'
FILEPATH_RESULTS = 'results/'
FILEPATH_V = 'results/value_functions/'
max_pi_iter = 10
multi_xyc = True
violin = True

theory_vs_sim = False
theory_discr_vs_solve = False
theory_vs_solve = False
vi_vs_solve = False
vi_vs_sim = False
ospi_vs_solve = False
solve_vs_sim = False
plot_pi_abs = False
plot_pi_rel = False

ids_to_analyze = {'J1': [1, 2, 3], 'J2': [1, 2, 3]}  # ID_i
summarize_policy = False
plot_policy = True
plot_g_mem = True
plot_v = True
plot_w = True
cap_d = 100
dep_arr = 0

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
if theory_vs_sim:
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
if theory_discr_vs_solve:
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
if theory_vs_solve:
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
if vi_vs_solve:
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
if vi_vs_sim:
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
if ospi_vs_solve:
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

# compare solve against sim per method
if solve_vs_sim:
    for method in methods_both:
        print(method)
        plotting.plot_gap(inst_set,
                          [method],
                          '_g_sim',
                          method,
                          '_g',
                          method,
                          '_g',
                          'Gap of g sims vs solve for ' + method,
                          multi_xyc=multi_xyc,
                          violin=violin)


# Diminishing opt. gap of policy iteration, abs. and rel. to OSPI
if (not is_sim) and (plot_pi_abs or plot_pi_rel):
    gap_abs = {'abs_' + str(i): [] for i in range(max_pi_iter + 1)}
    gap_rel = {'rel_' + str(i): [] for i in range(max_pi_iter)}
    for row in inst_set:
        g_mem = np.load(FILEPATH_V + '_'.join(['g', INSTANCES_ID, str(inst_id),
                                               'pi.npz']))['arr_0']
        gap_abs['abs_0'].append((row.vi_g - row.ospi_g) / row.vi_g)
        for i in range(max_pi_iter):
            g = g_mem[min(i, max_pi_iter)]
            gap_abs['abs_' + str(i+1)].append((row.vi_g - g) / row.vi_g * 100)
            gap_rel['rel_' + str(i)].append((row.ospi_g - g) / row.ospi_g * 100)
    if plot_pi_abs:
        plotting.multi_boxplot(gap_abs, gap_abs.keys,
                               'Policy Iteration opt. gap',
                               range(max_pi_iter + 1),
                               'gap (%)',
                               violin=violin)
    if plot_pi_rel:
        plotting.multi_boxplot(gap_rel, gap_rel.keys,
                               'Policy Iteration gap rel. to OSPI',
                               range(max_pi_iter),
                               'gap (%)',
                               violin=violin)

# Plotting Pi, V, W
for inst_id in ids_to_analyze[INSTANCES_ID]:
    inst = inst_set.iloc[inst_id]
    env = Env(J=inst.J, S=inst.S, D=inst.D,
              gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
              mu=inst.mu, lab=inst.lab)
    for method in solve_methods:
        name = method + ' inst: ' + INSTANCES_ID + '_' + str(inst_id)
        state = plotting.state_selection(env,
                                         dim_i=True,
                                         s=1,
                                         wait_perc=0.7)
        state_i = np.concatenate(([dep_arr], state))
        if plot_policy or summarize_policy:
            Pi = np.load(FILEPATH_V + '_'.join(['pi',
                                                INSTANCES_ID,
                                                str(inst_id),
                                                method + '.npz']))['arr_0']
            if summarize_policy:
                pi_learner = PolicyIteration(Pi=Pi)
                tools.summarize_policy(env, pi_learner, print_per_time=False)
            if plot_policy:
                plotting.plot_heatmap(env, state_i,
                                      Pi=Pi,
                                      title='PI ',
                                      t=inst.t,
                                      cap_d=cap_d)
        if plot_w:
            w = np.load(FILEPATH_V + '_'.join(['w',
                                               INSTANCES_ID,
                                               str(inst_id),
                                               method + '.npz']))['arr_0']
            state_w = np.concatenate(([0], state))
            plotting.plot_heatmap(env, state_i,
                                  W=w,
                                  title='W ',
                                  t=inst.t,
                                  cap_d=cap_d)
        if plot_v:
            V = np.load(FILEPATH_V + '_'.join(['v',
                                               INSTANCES_ID,
                                               str(inst_id),
                                               method + '.npz']))['arr_0']
            plotting.plot_heatmap(env, state_i,
                                  V=V,
                                  title='V ',
                                  t=inst.t,
                                  cap_d=cap_d)
    if plot_g_mem:
        g_mem = np.load(FILEPATH_V + '_'.join(['g', INSTANCES_ID, str(inst_id),
                                               'pi.npz']))['arr_0']
        g_ospi = np.load(FILEPATH_V + '_'.join(['g', INSTANCES_ID, str(inst_id),
                                               'ospi.npz']))['arr_0']
        plt.scatter(range(1 + len(g_mem)), [g_ospi] + g_mem)
        plt.xlabel('Iterations')
        plt.ylabel('g')
        plt.title('Policy Iteration starting with OSPI')
        plt.show()


# find file
# for file in os.listdir(FILEPATH_V):
#     if not (file.startswith('g_' + INSTANCES_ID)):
#         continue  # or file.endswith('_pi.npz')):
#     print(file)

# Old code
# exp_wait, g, success_prob = tools.get_gen_erlang_c(inst_row, 1e6)
# print(f'inst: {instance_name} \n'
#       f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
#       f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
#       f'P(W<t) = {["%.4f" % elem for elem in success_prob]}')
