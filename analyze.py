"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from utils import plotting, tools, TimeConstraintEDs as Env, PolicyIteration

FILEPATH_RESULTS = 'results/'
FILEPATH_V = 'results/value_functions/'

overview = False

instance_id = 'J1'
use_g_tmp = False
max_pi_iter = 10
multi_xyc = False
violin = False

theory_vs_sim = False
theory_discr_vs_solve = False
theory_vs_solve = False
vi_vs_solve = False
vi_vs_sim = False
ospi_vs_solve = False
solve_vs_sim = False

plot_pi_abs = False
plot_pi_rel = False

ids_to_analyze = {'J1': [107], 'J2': [47, 59]}  # ID_i
summarize_policy = True
calculate_pi = True
plot_policy = True
plot_g_mem = True
plot_v = True
plot_w = True
cap_d = 100
dep_arr = 0

analyze_gamma = False

g_tmp = '_g_tmp' if use_g_tmp else '_g'
methods_gam_analyze = ['vi', 'ospi']

# pd.set_option('display.max_columns', 12)
# pd.set_option('display.width', 200)

# find file
# for file in os.listdir(FILEPATH_V):
#     if not (file.startswith('g_' + INSTANCES_ID)):
#         continue  # or file.endswith('_pi.npz')):
#     print(file)

# give an overview of methods solved / unsolved
if overview:
    for instance_id_tmp in ['J1', 'J2', 'J2_D_gam',
                            'J1_sim', 'J2_sim', 'sim_sim']:
        instance_name_tmp = 'instances_' + instance_id_tmp
        inst_set_tmp = tools.inst_load(FILEPATH_RESULTS + instance_name_tmp +
                                       '.csv')
        is_sim = 'sim' in instance_id_tmp
        print(instance_name_tmp)
        if is_sim:
            tools.solved_and_left(inst_set_tmp, sim=True)
        else:
            for g_tmp_b in [False, True]:
                print('Using _g_tmp' if g_tmp_b else 'Using _g')
                tools.solved_and_left(inst_set_tmp, sim=False,
                                      use_g_tmp=g_tmp_b)
        print('\n')

# Load in instance_set and extract methods used
instance_name = 'instances_' + instance_id
inst_set = tools.inst_load(FILEPATH_RESULTS + instance_name + '.csv')
is_sim = 'sim' in instance_id
tools.solved_and_left(inst_set, sim=is_sim, use_g_tmp=use_g_tmp)
solve_methods = []
inst_set['smu(1-rho)'] = tools.get_smu_rho(inst_set.lab,
                                           inst_set.mu,
                                           inst_set.S)
if not is_sim:
    solve_methods = ['_'.join(column.split('_')[:-2])
                     for column in inst_set.columns
                     if column.endswith('job_id')]
    inst_set_gammas = inst_set.copy()
    inst_set = inst_set[inst_set['gamma'] == max(inst_set['gamma'])]

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
                      'Gap of g for sims vs theory for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' sim',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)
# theory discr. vs solve
if theory_discr_vs_solve:
    plotting.plot_gap(inst_theory,
                      solve_methods,
                      g_tmp,
                      'theory',
                      '_g_gam',
                      'theory',
                      '_g_gam',
                      'Gap of g for solve vs discr theory for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' solve & theory discr.',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)
# theory vs solve
if theory_vs_solve:
    plotting.plot_gap(inst_theory,
                      solve_methods,
                      g_tmp,
                      'theory',
                      '_g',
                      'theory',
                      '_g',
                      'Gap of g for solve vs theory for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' solve & theory',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)

# compare VI vs solve
if vi_vs_solve:
    plotting.plot_gap(inst_set,
                      solve_methods,
                      g_tmp,
                      'vi',
                      g_tmp,
                      'vi',
                      g_tmp,
                      'Gap of g for solve vs VI for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' solve vs VI',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)

# compare VI vs sim
if vi_vs_sim:
    plotting.plot_gap(inst_set,
                      sim_methods,
                      '_g_sim',
                      'vi',
                      g_tmp,
                      'vi',
                      g_tmp,
                      'Gap of g for sim vs VI for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' sim vs VI',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)

# compare OSPI vs solve (rel to vi)
if ospi_vs_solve:
    plotting.plot_gap(inst_set,
                      solve_methods,
                      g_tmp,
                      'ospi',
                      g_tmp,
                      'vi',
                      g_tmp,
                      'Gap of g for sim vs OSPI rel to VI for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' solve vs OSPI',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)

# compare solve against sim per method
if solve_vs_sim:
    for method in methods_both:
        print(method)
        plotting.plot_gap(inst_set,
                          [method],
                          '_g_sim',
                          method,
                          g_tmp,
                          method,
                          g_tmp,
                          'Gap of g, (solve_g - sims-g) for ' + method,
                          multi_xyc=multi_xyc,
                          title_xyc=instance_id + ' solve_g - sim_g',
                          violin=violin,
                          rotation=20,
                          left=0.1,
                          bottom=0.1)

# Diminishing opt. gap of policy iteration, abs. and rel. to OSPI
if (not is_sim) and (plot_pi_abs or plot_pi_rel):
    gap_abs = {'abs_' + str(i): [] for i in range(max_pi_iter)}
    gap_rel = {'rel_' + str(i): [] for i in range(max_pi_iter)}
    for i, row in inst_set.iterrows():
        g_mem = np.load(FILEPATH_V + '_'.join(['g', instance_id, str(i),
                                               'pi.npz']))['arr_0']
        vi_g, ospi_g = row['vi' + g_tmp], row['ospi' + g_tmp]
        for i in range(max_pi_iter):
            g = g_mem[min([i, max_pi_iter, len(g_mem) - 1])]
            if not np.isnan(vi_g):
                gap_abs['abs_' + str(i)].append((vi_g - g) / vi_g * 100)
            if not np.isnan(ospi_g):
                gap_rel['rel_' + str(i)].append((ospi_g - g) / ospi_g * 100)
    if plot_pi_abs:
        plotting.multi_boxplot(gap_abs, gap_abs.keys(),
                               'Policy Iteration opt. gap',
                               range(max_pi_iter),
                               'gap (%)',
                               violin=violin,
                               rotation=0)
    if plot_pi_rel:
        plotting.multi_boxplot(gap_rel, gap_rel.keys(),
                               'Policy Iteration gap rel. to OSPI',
                               range(max_pi_iter),
                               'gap (%)',
                               violin=violin,
                               rotation=0)

# Plotting Pi, V, W
if instance_id in ids_to_analyze:
    for inst_id in ids_to_analyze[instance_id]:
        if inst_id not in inst_set.index:
            continue
        inst = inst_set.loc[inst_id]
        env = Env(J=inst.J, S=inst.S, D=inst.D,
                  gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                  mu=inst.mu, lab=inst.lab)
        for method in ['vi', 'ospi']:
            name = method + ' inst: ' + instance_id + '_' + str(inst_id)
            state = plotting.state_selection(env,
                                             dim_i=True,
                                             s=1,
                                             wait_perc=0.7)
            state_i = np.concatenate(([dep_arr], state))
            file = '_'.join([instance_id, str(inst_id), method + '.npz'])
            file_pi = 'pi_' + file
            file_w = 'w_' + file
            file_v = 'v_' + file
            V = np.load(FILEPATH_V + file_v)['arr_0']
            Pi = None
            if (file_pi in os.listdir(FILEPATH_V)
                    and (plot_policy or summarize_policy)):
                Pi = np.load(FILEPATH_V + file_pi)['arr_0']
            elif calculate_pi and (plot_policy or summarize_policy):
                pi_learner = PolicyIteration()
                pi_learner.one_step_policy_improvement(env, V)
                Pi = pi_learner.Pi
                w = pi_learner.W
            if (Pi is not None) and summarize_policy:
                pi_learner = PolicyIteration(Pi=Pi)
                tools.summarize_policy(env, pi_learner, print_per_time=False)
            if (Pi is not None) and plot_policy:
                plotting.plot_heatmap(env, state_i,
                                      Pi=Pi,
                                      title=method + ' policy ',
                                      t=inst.t,
                                      cap_d=cap_d)
            if plot_w:
                if not calculate_pi:
                    w = np.load(FILEPATH_V + file_w)['arr_0']
                plotting.plot_heatmap(env, state_i,
                                      W=w,
                                      title=method + ' W ',
                                      t=inst.t,
                                      cap_d=cap_d)
            # if file_pi in os.listdir(FILEPATH_V) and
            if plot_v:
                plotting.plot_heatmap(env, state,
                                      V=V,
                                      title=method + ' V ',
                                      t=inst.t,
                                      cap_d=cap_d)
        if plot_g_mem:
            g_mem = np.load(FILEPATH_V + '_'.join(
                ['g', instance_id, str(inst_id), 'pi.npz']))['arr_0']
            fig, ax = plt.subplots(1)
            ax.scatter(range(len(g_mem)), g_mem)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('g')
            ax.set_title('Policy Iteration starting with OSPI')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.show()

# Analyze different gamma
if analyze_gamma and len(set(inst_set_gammas['gamma'])) > 1:
    # note that size and smu(1-rho) are just for plotting, not for key
    key_cols = ['J', 'S', 'mu', 'load', 'imbalance', 'smu(1-rho)']
    cols = [*key_cols, 'gamma', 'D', *[x + g_tmp for x in methods_gam_analyze]]
    inst_set_tmp = inst_set_gammas[cols].copy()
    inst_set_tmp['mu'] = inst_set_tmp['mu'].apply(str)  # np array unhashable
    inst_set_tmp['imbalance'] = inst_set_tmp['imbalance'].apply(str)
    data = inst_set_tmp[key_cols].copy().drop_duplicates(ignore_index=True)
    gammas = sorted(list(set(inst_set_tmp['gamma'])))
    if instance_id == 'J2_D_gam':
        d_multiples = [5, 10, 0]  # 0 must be last
    elif instance_id == 'J3':
        d_multiples = [4, 0]
    else:
        d_multiples = [0]
    methods = []
    for gamma in gammas:
        for d in d_multiples:
            for method in methods_gam_analyze:
                name = method + '_gam_' + str(gamma) + '_' + str(d)
                methods.append(name)
                if d > 0:
                    mask = ((inst_set_tmp.D == int(d * gamma)) &
                            (inst_set_tmp.gamma == gamma))
                else:
                    # Note that duplicates are removed, as checking for
                    # ~D isin(gamma*d) does not work for duplicates
                    mask = (inst_set_tmp.gamma == gamma)
                data_slice = inst_set_tmp[mask].copy()
                data_slice.rename(columns={method + g_tmp: name + g_tmp},
                                  inplace=True)
                drop_methods = [m + g_tmp for m in methods_gam_analyze
                                if m != method]
                data_slice.drop(columns=['gamma', 'D', *drop_methods],
                                inplace=True)
                # Drop duplicates to ensure that the merge is correct
                # Does not matter which to remove, as it are duplicates
                data_slice = data_slice[~data_slice[key_cols].duplicated()]
                data = pd.merge(data, data_slice, on=key_cols, how='left')
            # Remove duplicates from slicing set, to be left with
            inst_set_tmp.drop(data_slice.index, inplace=True)
    methods.sort()  # to get logical order in picture
    ref_m = 'vi_gam_' + str(max(gammas)) + '_0'
    plotting.plot_gap(data,
                      methods,
                      g_tmp,
                      ref_m, g_tmp,
                      ref_m, g_tmp,
                      'Gap of g for different D and gamma for ' + instance_id,
                      multi_xyc=False,
                      violin=violin,
                      x_lab='imbalance',
                      rotation=80,
                      left=0.1,
                      bottom=0.28)  # 0.27 for J2_D_gam

# Old code
# exp_wait, g, success_prob = tools.get_gen_erlang_c(inst_row, 1e6)
# print(f'inst: {instance_name} \n'
#       f'upper bound of g: {sum(inst.r[0] * inst.lab[0]):0.4f} \n'
#       f'Theory, g={g:0.4f}, E(W)={exp_wait:0.4f}, '
#       f'P(W<t) = {["%.4f" % elem for elem in success_prob]}')

# if vi but missing pi
# pi_learner = PolicyIteration()
# v_file = ('v_' + args.instance + '_' + str(inst.iloc[0]) + '_vi.npz')
# if v_file in os.listdir(FILEPATH_V):
    # learner.V = np.load(FILEPATH_V + v_file)['arr_0']
    # env.convergence_check = 1
    # env.max_iter = 1
    # learner.value_iteration(env, pi_learner)

# if remove_unsolved:
#     for method in ['vi']:  # solve_methods
#         inst_set = inst_set[~inst_set[method + g_tmp].isna()]