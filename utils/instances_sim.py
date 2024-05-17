import numpy as np
from utils import tools, TimeConstraintEDs as Env


def generate_instance(inst, sim_id):
    """
    For every simulation parameter set, generate instance file with all methods.
    :param inst: Empty dataframe with appropriate columns and a row per method.
    :param sim_id: ID of the simulation parameter set.
    :return: instance dataframe file filled with parameters
    """

    inst['gamma'] = 10
    inst['D'] = 180 * 10  # max time * gamma
    if sim_id in [1, 2, 3, 4, 5, 6]:
        inst['J'] = 3
        inst['S'] = 5
        inst['r'] = [np.array([1] * 3) for _ in range(len(inst))]
        inst['c'] = [np.array([1] * 3) for _ in range(len(inst))]
        inst['imbalance'] = [np.array([18, 94, 172]) / 284 for r in
                             range(len(inst))]
        inst['load'] = 0.85 if sim_id == 1 else 0.95
        if sim_id in [4, 5, 6]:
            inst['t'] = [np.array([60] * 3) for _ in range(len(inst))]
        else:
            inst['t'] = [np.array([10, 60, 120]) for _ in range(len(inst))]
        if sim_id in [3, 5, 6]:
            inst['mu'] = [np.array([1] * 3) / 60 for _ in range(len(inst))]
        else:
            inst['mu'] = [np.array([1/2.19, 1/2, 1/0.51]) / 60 for _ in
                          range(len(inst))]
        if sim_id == 6:
            inst['imbalance'] = [np.array([1/3] * 3) for _ in range(len(inst))]
    elif sim_id in [7, 8]:
        inst['J'] = 4
        inst['S'] = 5
        inst['t'] = [np.array([60] * 4) for _ in range(len(inst))]
        inst['c'] = [np.array([1, 1, 0.5, 0.5]) for _ in range(len(inst))]
        inst['r'] = [np.array([1] * 4) for _ in range(len(inst))]
        inst['mu'] = [np.array([1, 2, 1, 2]) / 60 for _ in range(len(inst))]
        inst['imbalance'] = [np.array([1/4, 1/4, 3/4, 3/4]) / 2
                             for _ in range(len(inst))]
        inst['load'] = 0.9 if sim_id == 7 else 0.95  # if sim_id == 8
    elif sim_id in [9, 10]:
        inst['J'] = 6
        inst['S'] = 10
        inst['t'] = [np.array([60] * 6) for _ in range(len(inst))]
        inst['c'] = [np.array([1] * 6) for _ in range(len(inst))]
        inst['r'] = [np.array([1] * 6) for _ in range(len(inst))]
        inst['mu'] = [np.arange(1, 7) for _ in range(len(inst))]
        inst['imbalance'] = [np.arange(1, 7) / 21 for r in
                             range(len(inst))]
        inst['load'] = 0.9 if sim_id == 9 else 0.95  # if sim_id == 10
    elif sim_id == 11:
        inst['J'] = 1
        inst['S'] = 5
        inst['t'] = [np.array([60]) for _ in range(len(inst))]
        inst['c'] = [np.array([1]) for _ in range(len(inst))]
        inst['r'] = [np.array([1]) for _ in range(len(inst))]
        inst['mu'] = [np.array([1/30]) for _ in range(len(inst))]
        inst['load'] = 0.85
        inst['imbalance'] = [np.array([1]) for _ in range(len(inst))]
    if sim_id in [12, 13, 14]:
        row = [8, 57, 93][[12, 13, 14].index(sim_id)]
        inst_vi = tools.inst_load('results/instances_01.csv')
        for c_name in ['J', 'S', 'D', 'gamma', 'load']:
            inst[c_name] = inst_vi.loc[row][c_name]
        for c_name in ['t', 'c', 'r', 'mu', 'lab']:
            inst[c_name] = [inst_vi.loc[row][c_name] for _ in range(len(inst))]
    else:
        env = Env(J=inst['J'][0], S=inst['S'][0], D=inst['D'][0],
                  gamma=inst['gamma'][0],
                  t=inst['t'][0], c=inst['c'][0], r=inst['r'][0],
                  mu=inst['mu'][0],
                  load=inst['load'][0], imbalance=inst['imbalance'][0],
                  sim=True)
        inst['lab'] = [env.lab for _ in range(len(inst))]
    return inst
