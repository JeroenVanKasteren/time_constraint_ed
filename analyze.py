"""
Load and visualize results.

@author: Jeroen van Kasteren (jeroen.van.kasteren@vu.nl)
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from utils import (plotting, tools, TimeConstraintEDs as Env, PolicyIteration,
                   OneStepPolicyImprovement)

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

ids_to_analyze = {'J1': list(range(27, 28)), 'J2': [47, 59]}  # ID_i
ref_method = 'vi'
comp_methods = ['vi', 'ospi', 'pi', 'fcfs']
summarize_policy = False
tol = 1e-1
check_v_app = True
print_states_pi = True
print_states_v = False
print_states_v_app = False
calculate_pi = True
plot_policy = True
plot_g_mem = False
plot_v = False
plot_w = False
cap_d = 100
dep_arr = 0
dec = 4
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

# add isolated g
inst_set['isolated_g'] = 0.0
for i, inst in inst_set.iterrows():
    env_i = Env(J=inst.J, S=inst.S, D=inst.D,
                gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                mu=inst.mu, lab=inst.lab)
    inst_set.loc[i, 'isolated_g'] = sum(env_i.g)

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
inst_mm1 = inst_set[inst_set.apply(lambda x_row: len(set(x_row.mu)) == 1,
                                   axis=1)].reindex()
mm1_g = []
mm1_discr_g = []
for i, inst in inst_mm1.iterrows():
    # M/M/1 g
    prob_i = inst.lab/sum(inst.lab)
    _, _, _, prob_late = tools.get_erlang_c(inst.mu[0], inst.load, inst.S,
                                            inst.t, prob_i)
    mm1_g.append(sum(inst.lab * (inst.r - inst.c * prob_late)))
    # M/M/1 discr g
    env = Env(J=1, S=inst.S, D=inst.D,
              gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
              mu=[sum(inst.lab) / sum(sum(inst.lab) / inst.mu)],
              lab=[sum(inst.lab)])
    mm1_discr_g.append(sum(env.g))
    # optional test:
    # _, g, _ = tools.get_erlang_c_gam(row, 1e6)
    # mm1_g.append(g)
inst_mm1.loc[:, 'mm1_g'] = mm1_g
inst_mm1.loc[:, 'mm1_discr_g'] = mm1_discr_g

# MM1 vs sim
if theory_vs_sim:
    plotting.plot_gap(inst_mm1,
                      sim_methods,
                      '_g_sim',
                      'mm1',
                      '_g',
                      'mm1',
                      '_g',
                      'Gap of g for sims vs M/M/1 for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' sim',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)
# MM1 discr. vs solve
if theory_discr_vs_solve:
    plotting.plot_gap(inst_mm1,
                      solve_methods,
                      g_tmp,
                      'mm1_discr',
                      '_g',
                      'mm1_discr',
                      '_g',
                      'Gap of g for solve vs M/M/1 discr for ' + instance_id,
                      multi_xyc=multi_xyc,
                      title_xyc=instance_id + ' solve & M/M/1 discr',
                      violin=violin,
                      rotation=20,
                      left=0.1,
                      bottom=0.1)
# MM1 vs solve
if theory_vs_solve:
    plotting.plot_gap(inst_mm1,
                      solve_methods,
                      g_tmp,
                      'mm1',
                      '_g',
                      'mm1',
                      '_g',
                      'Gap of g for solve vs M/M/1 for ' + instance_id,
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
    gap_abs = {'abs_' + str(i): [] for i in range(max_pi_iter + 1)}
    gap_rel = {'rel_' + str(i): [] for i in range(max_pi_iter + 1)}
    gap_imp_abs = {'abs_' + str(i): [] for i in range(max_pi_iter)}
    gap_imp_rel = {'rel_' + str(i): [] for i in range(max_pi_iter)}
    for i, inst in inst_set.iterrows():
        g_mem = np.load(FILEPATH_V + '_'.join(['g', instance_id, str(i),
                                               'pi.npz']))['arr_0']
        vi_g, ospi_g = inst['vi' + g_tmp], inst['ospi' + g_tmp]
        # add isolated g
        env_i = Env(J=inst.J, S=inst.S, D=inst.D,
                    gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                    mu=inst.mu, lab=inst.lab)
        if not np.isnan(vi_g):
            gap_abs['abs_0'].append((vi_g - sum(env_i.g)) / vi_g * 100)
        if not np.isnan(ospi_g):
            gap_rel['rel_0'].append((ospi_g - sum(env_i.g)) / ospi_g * 100)
        for j in range(1, max_pi_iter + 1):
            g = g_mem[min([j, max_pi_iter, len(g_mem) - 1])]
            if not np.isnan(vi_g):
                gap_abs['abs_' + str(j)].append((vi_g - g) / vi_g * 100)
                gap_imp_abs['abs_' + str(j-1)].append(
                    gap_abs['abs_' + str(j-1)][-1]
                    - gap_abs['abs_' + str(j)][-1])
            if not np.isnan(ospi_g):
                gap_rel['rel_' + str(j)].append((ospi_g - g) / ospi_g * 100)
                gap_imp_rel['rel_' + str(j-1)].append(
                    gap_rel['rel_' + str(j-1)][-1]
                    - gap_rel['rel_' + str(j)][-1])
    if plot_pi_abs:
        plotting.multi_boxplot(gap_abs, gap_abs.keys(),
                               'Policy Iteration opt. gap',
                               range(max_pi_iter + 1),
                               'gap (%)',
                               violin=violin,
                               rotation=0)
        plotting.multi_boxplot(gap_imp_abs, gap_imp_abs.keys(),
                               'Policy Iteration improvement in opt. gap',
                               range(1, max_pi_iter + 1),
                               'Improvement in gap-%',
                               violin=violin,
                               rotation=0)
    if plot_pi_rel:
        plotting.multi_boxplot(gap_rel, gap_rel.keys(),
                               'Policy Iteration gap rel. to OSPI',
                               range(max_pi_iter + 1),
                               'gap (%)',
                               violin=violin,
                               rotation=0)
        plotting.multi_boxplot(gap_imp_abs, gap_imp_abs.keys(),
                               'Policy Iteration improvement in '
                               'gap rel. to OSPI',
                               range(1, max_pi_iter + 1),
                               'Improvement in gap-%)',
                               violin=violin,
                               rotation=0)

# Analyzing & Plotting Pi, V, W
if instance_id in ids_to_analyze:
    for inst_id in ids_to_analyze[instance_id]:
        if inst_id not in inst_set.index:
            continue
        inst = inst_set.loc[inst_id]
        env = Env(J=inst.J, S=inst.S, D=inst.D,
                  gamma=inst.gamma, t=inst.t, c=inst.c, r=inst.r,
                  mu=inst.mu, lab=inst.lab)
        dep_arr = inst.J
        zero_state = tuple(np.zeros(len(env.dim), dtype=int))

        # load in ref V, w, Pi, v_app
        file = '_'.join([instance_id, str(inst_id), ref_method + '.npz'])
        g_ref = round(inst[ref_method + g_tmp], dec)
        V_ref = np.load(FILEPATH_V + 'v_' + file)['arr_0']
        Pi_ref = None
        if 'pi_' + file in os.listdir(FILEPATH_V):
            Pi_ref = np.load(FILEPATH_V + 'pi_' + file)['arr_0']
        elif calculate_pi:
            pi_learner = PolicyIteration()
            pi_learner.one_step_policy_improvement(env, V)
            Pi_ref = pi_learner.Pi
            w_ref = pi_learner.W

        print('\n')
        if g_ref is None:
            print('reference g_' + method + ' stuck')
        if check_v_app:
            learner_ospi = OneStepPolicyImprovement()
            v_app = learner_ospi.get_v_app(env)
            mask = ~np.isclose(v_app - v_app[zero_state],
                               V_ref - V_ref[zero_state],
                               rtol=1e-3)
            print('V_app equal to ref?', not mask.any())
            if print_states_v_app:
                print(np.transpose(mask.nonzero()))

        print('Instance', instance_id, 'id', inst_id,
              'ref_method', ref_method, '\n')

        for method in comp_methods:
            # load in data
            file = '_'.join([instance_id, str(inst_id), method + '.npz'])
            g = round(inst[method + g_tmp], dec)
            V = np.load(FILEPATH_V + 'v_' + file)['arr_0']
            Pi = None
            if ('pi_' + file in os.listdir(FILEPATH_V)
                    and (plot_policy or summarize_policy)):
                Pi = np.load(FILEPATH_V + 'pi_' + file)['arr_0']
            elif calculate_pi and (plot_policy or summarize_policy):
                pi_learner = PolicyIteration()
                pi_learner.one_step_policy_improvement(env, V)
                Pi = pi_learner.Pi
                w = pi_learner.W

            if g is None:
                print('g_' + method + ' stuck')
            print('g_' + method + ': ', g, ' equal to ref?', g == g_ref)

            # Summarize policy
            if (Pi is not None) and summarize_policy:
                pi_learner = PolicyIteration(Pi=Pi)
                tools.summarize_policy(env, pi_learner, print_per_time=False)

            name = method + ' inst: ' + instance_id + '_' + str(inst_id)
            # Plotting policy
            state = plotting.state_selection(env,
                                             dim_i=True,
                                             s=1,
                                             wait_perc=0.7)
            state_i = np.concatenate(([dep_arr], state))
            if (Pi is not None) and plot_policy:
                plotting.plot_heatmap(env, state_i,
                                      Pi=Pi,
                                      title=name + ' policy ',
                                      t=inst.t * inst.gamma,
                                      cap_d=cap_d)
            if plot_w:
                if not calculate_pi:
                    w = np.load(FILEPATH_V + 'w_' + file)['arr_0']
                plotting.plot_heatmap(env, state_i,
                                      W=w,
                                      title=name + ' W ',
                                      t=inst.t,
                                      cap_d=cap_d)
            # if file_pi in os.listdir(FILEPATH_V) and
            if plot_v:
                plotting.plot_heatmap(env, state,
                                      V=V,
                                      title=name + ' V ',
                                      t=inst.t,
                                      cap_d=cap_d)
            # comparing pi to reference
            if (Pi is not None) and (Pi_ref is not None):
                mask = ~np.isclose(Pi, Pi_ref, rtol=tol)
                print('Policy of ' + method + ' equal to ref?', not mask.any())
                if print_states_pi:
                    print(np.transpose(mask.nonzero()))
            # comparing V to reference
            mask = ~np.isclose(V - V[zero_state],
                               V_ref - V_ref[zero_state],
                               rtol=tol)
            print('V of ' + method + ' equal to ref?', not mask.any())
            if print_states_v:
                print(np.transpose(mask.nonzero()))
            # comparing V to V_app
            if check_v_app:
                mask = ~np.isclose(v_app - v_app[zero_state],
                                   V - V[zero_state],
                                   rtol=tol)
                print('V_app equal to V of ' + method + '? ', not mask.any())
                if print_states_v_app:
                    print(np.transpose(mask.nonzero()))
            print('\n')
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
