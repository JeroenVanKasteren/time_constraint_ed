"""
Static functions for the project.
"""

import numpy as np
import pandas as pd
from time import strptime
from sklearn.model_selection import ParameterGrid
from utils import TimeConstraintEDs as Env


def def_sizes(dim):
    """Docstring."""
    sizes = np.zeros(len(dim), np.int32)
    sizes[-1] = 1
    for i in range(len(dim) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dim[i + 1]
    return sizes


def strip_split(x):
    return np.array([float(i) for i in x.strip('[]').split(', ')])


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
            days, time = 0, '0:'+time_string
        else:
            days, time = 0, time_string
        x = strptime(time, '%H:%M:%S')
        return (((int(days) * 24 + x.tm_hour) * 60 + x.tm_min) * 60
                + x.tm_sec - 60)
    else:
        return np.Inf


def sec_to_time(time):
    """Convert seconds to minutes and return readable format."""
    time = int(time)
    if time >= 60*60:
        return f"(HH:MM:SS): {time // (60*60):02d}:{(time // 60) % 60:02d}:{time % 60:02d}"
    else:
        return f"(MM:SS): {time // 60:02d}:{time % 60:02d}"


def get_instance_grid(J, gamma, e, P, t, c, r, param_grid, max_target_prob):
    grid = pd.DataFrame(ParameterGrid(param_grid))
    print("Length of grid:", len(grid))

    grid['J'] = J
    grid['gamma'] = gamma
    grid['e'] = e
    grid['P'] = P

    grid['target_prob'] = 0
    grid['D'] = [[0]*J]*len(grid)
    grid['size'] = 0
    grid['size_i'] = 0
    grid['mu'] = [[] for r in range(len(grid))]
    grid['lab'] = [[] for r in range(len(grid))]
    grid['t'] = [[] for r in range(len(grid))]
    grid['c'] = [[] for r in range(len(grid))]
    grid['r'] = [[] for r in range(len(grid))]

    for i, inst in grid.iterrows():
        env = Env(J=J, S=inst.S, gamma=gamma, P=P, e=e, t=t, c=c, r=r,
                  mu=np.array([inst.mu_1, inst.mu_2]),  # TODO: generalize
                  load=inst.load,
                  imbalance=np.array([inst.imbalance, 1]))
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


def update_mean(mean, x, n):
    """Welford's method to update the mean. Can be set to numba function."""
    return mean + (x - mean) / n  # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__