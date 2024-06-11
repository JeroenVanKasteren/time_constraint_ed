"""
Static functions for the project.
"""

import argparse
import numba as nb
import numpy as np
import os
import pandas as pd
import pickle as pkl
from scipy.stats import norm
from scipy.special import factorial as fac
from sklearn.model_selection import ParameterGrid
from utils import TimeConstraintEDs as Env
from utils import OneStepPolicyImprovement as Ospi


def conf_int(alpha, data):
    return norm.ppf(1 - alpha / 2) * data.std() / np.sqrt(len(data))


def def_sizes(dim):
    """Docstring."""
    sizes = np.zeros(len(dim), np.int32)
    sizes[-1] = 1
    for i in range(len(dim) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dim[i + 1]
    return sizes


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def fixed_order(env, method):
    """
    Return order of method

    :param: env: dictenvironment
    :param method: method to sort by (cmu_t_min, cmu_t_max, l_min, l_max)
    :type method: str
    """
    # lexsort: Sort ascending from last to first entry (negate for descending)
    if method == 'cmu_t_min':
        return np.lexsort([np.arange(env.J), env.t, -env.c * env.mu])
    elif method == 'cmu_t_max':
        return np.lexsort([-np.arange(env.J), -env.t, -env.c * env.mu])
    elif method == 'l_max':
        return np.lexsort([-np.arange(env.J), -env.lab])
    elif method == 'l_min':
        return np.lexsort([np.arange(env.J), env.lab])


def generate_times(env, n):
    """Generate exponential arrival and service times."""
    arrival_times = nb.typed.List[np.float32]()  # +1, last arrival
    service_times = nb.typed.List[np.float32]()
    for i in range(env.J):
        arrival_times.append(nb.typed.List(env.rng.exponential(1 / env.lab[i],
                                                               n + env.J)))
        service_times.append(nb.typed.List(env.rng.exponential(1 / env.mu[i],
                                                               n + env.J)))
    return arrival_times, service_times


def get_erlang_c(mu, rho, s, t=0):
    """
    Calculates Erlang C statistics

    parameters
        mu  (float): service rate
        rho (float): load
        s   (int): number of servers
        [t  (int): target t]

    return
        erlang_c  (float): probability of delay, P(W>0)
        exp_w     (float): expected waiting time, E(W_q)
        pi_0      (float): P(x = 0), none idle and no one waiting
        prob_late (float): P(W>t)
    """

    lab = s * mu * rho
    a = s * rho

    j = np.arange(s)  # sum_{j=0}^{s-1}
    trm = a ** s / (fac(s - 1) * (s - a))

    pi_0 = 1 / (sum(a ** j / fac(j)) + trm)
    erlang_c = trm * pi_0
    exp_w = erlang_c / (s * mu - lab)
    prob_late = erlang_c * np.exp(-(s * mu - lab) * t) if t >= 0 else 0
    return erlang_c, exp_w, pi_0, prob_late


def get_instance_grid(J, e, P, t, c, r, param_grid, max_target_prob,
                      gamma_multi=0):
    grid = pd.DataFrame(ParameterGrid(param_grid))
    print("Length of grid:", len(grid))

    grid['J'] = J
    grid['e'] = e
    grid['P'] = P

    grid['target_prob'] = 0
    grid['D'] = [[0] * J] * len(grid)
    grid['size'] = 0
    grid['size_i'] = 0
    grid['mu'] = [[] for r in range(len(grid))]
    grid['lab'] = [[] for r in range(len(grid))]
    grid['t'] = [[] for r in range(len(grid))]
    grid['c'] = [[] for r in range(len(grid))]
    grid['r'] = [[] for r in range(len(grid))]

    for i, inst in grid.iterrows():
        if gamma_multi == 0:
            env = Env(J=J, S=inst.S, gamma=inst.gamma, P=P, e=e, t=t, c=c, r=r,
                      mu=np.array([inst.mu_1, inst.mu_2]),
                      load=inst.load,
                      imbalance=np.array([inst.imbalance, 1]))
        else:
            env = Env(J=J, S=inst.S, gamma=inst.gamma, P=P, e=e, t=t, c=c, r=r,
                      mu=np.array([inst.mu_1, inst.mu_2]),
                      load=inst.load,
                      imbalance=np.array([inst.imbalance, 1]),
                      D=int(inst.gamma*gamma_multi))
        grid.loc[i, 'target_prob'] = env.target_prob
        grid.loc[i, 'D'] = env.D
        grid.loc[i, 'size'] = env.size
        grid.loc[i, 'size_i'] = env.size_i
        for j in range(J):
            grid.loc[i, 'mu'].append(env.mu[j])
            grid.loc[i, 'lab'].append(env.lab[j])
            grid.loc[i, 't'].append(env.t[j])
            grid.loc[i, 'c'].append(env.c[j])
            grid.loc[i, 'r'].append(env.r[j])
    print('Removed instances due to target_prob > ', max_target_prob, ':',
          grid[grid['target_prob'] > max_target_prob])
    grid = grid[grid['target_prob'] < max_target_prob]
    return grid


def get_time(time_string):
    """Read in time in formats (D)D-HH:MM:SS, (H)H:MM:SS, or (M)M:SS.
    Format in front of time is removed."""
    if ((time_string is not None) & (not pd.isnull(time_string)) &
            (time_string != np.inf)):
        if '): ' in time_string:  # if readable format
            time_string = time_string.split('): ')[1]
        if '-' in time_string:
            days, time = time_string.split('-')
        elif time_string.count(':') == 1:
            days, time = 0, '0:' + time_string
        else:
            days, time = 0, time_string
        hour, minutes, sec = [int(x) for x in time.split(':')]
        return (((int(days) * 24 + hour) * 60 + minutes) * 60 + sec - 60)
    else:
        return np.Inf


def get_v_app(env):
    """Get the approximate value function for a given state."""
    v = np.zeros((env.J, env.D + 1))
    for i in range(env.J):
        v[i,] = Ospi.get_v_app_i(env, i)
    return v


def inst_load(filepath):
    cols = ['t', 'c', 'r', 'lab', 'mu']
    inst = pd.read_csv(filepath)
    inst.loc[:, cols] = inst.loc[:, cols].applymap(strip_split)
    return inst


def load_result(method, instance_name):
    inst = inst_load('results/' + instance_name)
    inst_id = instance_name[-6:-4]
    methods = inst['method'].values
    if isinstance(method, str):
        row_id = np.where(methods == method)[0][0]
    else:
        row_id = method
        method = inst['method'].values[row_id]
    file = 'result_' + inst_id + '_' + method + '.pkl'
    return (method, row_id, inst,
            pkl.load(open('results/simulation_pickles/' + file, 'rb')))


def load_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', default='0')  # SULRM_JOBID
    parser.add_argument('--array_id', default='0')  # SLURM_ARRAY_TASK_ID
    parser.add_argument('--time')  # SLURM_TIMELIMIT
    parser.add_argument('--instance', default='01')  # User input
    parser.add_argument('--method', default='not_specified')  # User input
    parser.add_argument('--x', default=0)  # User input
    parser.add_argument('--max_iter', default=np.Inf)  # User input
    args = parser.parse_args(raw_args)
    args.job_id = int(args.job_id)
    args.array_id = int(args.array_id)
    args.x = int(float(args.x))
    return args


def moving_average(kpi_df, k, m, t):
    times = kpi_df['time'].values[k + t - 1::t] - kpi_df['time'].values[k::t]
    return (kpi_df['reward'].values[k:].reshape(-1, m).sum(axis=0) / times,
            times)


def moving_average_admission(kpi_df, k, m):
    return kpi_df['reward'].values[k:].reshape(-1, m).mean(axis=0)


def remove_empty_files(directory):
    for file in os.listdir(directory):
        if os.path.getsize(os.path.join(directory, file)) == 0:
            os.remove(os.path.join(directory, file))


def round_significance(x, digits=1):
    return 0 if x == 0 else np.round(x, -int(np.floor(np.log10(abs(x)))) -
                                     (-digits + 1))


def sec_to_time(time):
    """Convert seconds to minutes and return readable format."""
    time = int(time)
    if time >= 60 * 60:
        return (f"(HH:MM:SS): {time // (60 * 60):02d}:{(time // 60) % 60:02d}:"
                f"{time % 60:02d}")
    else:
        return f"(MM:SS): {time // 60:02d}:{time % 60:02d}"


def solved_and_left(inst):
    methods = [column.split('_')[0] for column in inst.columns
               if column.endswith('job_id')]
    for method in methods:
        print('Solved ' + method + ': ' + str(inst[method + '_g'].count()) +
              ', left: ' +
              str(len(inst) - inst[method + '_g'].count()))
        if method != 'vi':
            print('Solved both for ' + method + ': ' +
                  str(inst[method + '_opt_gap'].count()))


def strip_split(x):
    if ',' in x:
        return np.array([float(i) for i in x.strip('[]').split(', ')])
    else:
        return np.array([float(i) for i in x.strip('[]').split()])


def update_mean(mean, x, n):
    """Welford's method to update the mean. Can be set to numba function."""
    return mean + (x - mean) / n  # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n


def summarize_policy(env, learner):
    unique, counts = np.unique(learner.Pi, return_counts=True)
    counts = counts / sum(counts)
    policy = pd.DataFrame(np.asarray((unique, counts)).T, columns=['Policy',
                                                                   'Freq'])
    policy.Policy = ['Keep Idle' if x == env.KEEP_IDLE
                     else 'None Waiting' if x == env.NONE_WAITING
                     else 'Servers Full' if x == env.SERVERS_FULL
                     else 'Invalid State' if x == env.NOT_EVALUATED
                     else 'Class ' + str(x) for x in policy.Policy]
    print(learner.name, 'g=', np.around(learner.g, int(-np.log10(env.e) - 1)))
    print(policy.to_string(formatters={'Freq': '{:,.2%}'.format}))

    feature_list = ['Class_' + str(i + 1) for i in range(env.J)]
    feature_list.extend(['Keep_Idle', 'None_Waiting', 'Servers_Full',
                         'Invalid_State'])
    counts = pd.DataFrame(0, index=np.arange(env.D), columns=feature_list)
    for x in range(env.D):
        states = [slice(None)] * (1 + env.J * 2)
        states[1] = x
        counts.loc[x, 'None_Waiting'] = np.sum(learner.Pi[tuple(states)]
                                               == env.NONE_WAITING)
        counts.loc[x, 'Servers_Full'] = np.sum(learner.Pi[tuple(states)]
                                               == env.SERVERS_FULL)
        counts.loc[x, 'Invalid_State'] = np.sum(learner.Pi[tuple(states)]
                                                == env.NOT_EVALUATED)
        total = np.prod(learner.Pi[tuple(states)].shape)
        total_valid = (total - counts.None_Waiting[x] - counts.Servers_Full[x]
                       - counts.Invalid_State[x])
        total_valid = 1 if total_valid == 0 else total_valid
        for i in range(env.J):
            states = [slice(None)] * (env.J * 2)
            states[1 + i] = x
            counts.loc[x, 'Class_' + str(i + 1)] = np.sum(
                learner.Pi[tuple(states)]
                == i + 1) / total_valid
        counts.loc[x, 'Keep_Idle'] = (np.sum(learner.Pi[tuple(states)]
                                             == env.KEEP_IDLE)
                                      / total_valid)
    counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] = \
        counts[['None_Waiting', 'Servers_Full', 'Invalid_State']] / total
    print(counts.to_string(formatters={'Class_1': '{:,.2%}'.format,
                                       'Class_2': '{:,.2%}'.format,
                                       'Keep_Idle': '{:,.2%}'.format,
                                       'None_Waiting': '{:,.2%}'.format,
                                       'Servers_Full': '{:,.2%}'.format,
                                       'Invalid_State': '{:,.2%}'.format}))
