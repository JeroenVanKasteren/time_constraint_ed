import numpy as np
from utils import tools, TimeConstraintEDs as Env


def generate_instance(sim_id):
    """
    For every simulation parameter set, generate instance file with all methods.
    :param sim_id: ID of the simulation parameter set.
    :return: param_grid (dict)
    """
    grid = {'gamma': 10,
            'D': 180 * 10}  # max time * gamma
    if sim_id in [1, 2, 3, 4, 5, 6]:
        J = 3
        grid['S'] = 5
        grid['imbalance'] = np.array([18, 94, 172]) / 284
        grid['load'] = 0.7 if sim_id == 1 else 0.85
        if sim_id in [4, 5, 6]:
            grid['t'] = np.array([60] * 3)
        else:
            grid['t'] = np.array([10, 60, 120])
        if sim_id in [3, 5, 6]:
            grid['mu'] = np.array([3] * 3) / 60
        else:
            grid['mu'] = np.array([2.19, 2, 0.51]) / 60
        if sim_id == 6:
            grid['imbalance'] = np.array([1/3] * 3)
    elif sim_id in [7, 8]:
        J = 4
        grid['S'] = 5
        grid['t'] = np.array([60] * 4)
        grid['c'] = np.array([1, 1, 0.5, 0.5])
        grid['mu'] = np.array([3, 2, 3, 2]) / 60
        grid['imbalance'] = np.array([1/4, 1/4, 3/4, 3/4]) / 2
        grid['load'] = 0.7 if sim_id == 7 else 0.85  # if sim_id == 8
    elif sim_id in [9, 10]:
        J = 6
        grid['S'] = 5
        grid['t'] = np.array([60] * 6)
        grid['mu'] = np.arange(3, 9) / 60
        grid['imbalance'] = np.arange(1, 7) / 21
        grid['load'] = 0.7 if sim_id == 9 else 0.85  # if sim_id == 10
    grid['J'] = J
    grid['lab'] = Env.get_lambda(grid['mu'], grid['S'],
                                 grid['load'], grid['imbalance'])
    return grid
